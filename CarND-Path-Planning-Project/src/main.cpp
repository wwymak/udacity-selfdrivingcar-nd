#include <fstream>
#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "json.hpp"
#include "spline.h"

using namespace std;

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

const double max_speed = 49;
int lane = 1;

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
	auto found_null = s.find("null");
	auto b1 = s.find_first_of("[");
	auto b2 = s.find_first_of("}");
	if (found_null != string::npos) {
		return "";
	} else if (b1 != string::npos && b2 != string::npos) {
		return s.substr(b1, b2 - b1 + 2);
	}
	return "";
}

double distance(double x1, double y1, double x2, double y2)
{
	return sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}

//convert mph (from the simulator) to meters per second (for the calcuations)
double mileph2meterps(double miles) {return miles * 0.44704;}
//conversions the otherway
double meterps2mileph(double meters) {return meters / 0.44704;}

vector<double> convert2GlobalCoords(vector<double> carCoords, double car_yaw) {
    double local_x = carCoords.at(0);
    double local_y = carCoords.at(1);

    double global_x = local_x * cos(car_yaw) - local_y * sin(car_yaw);
    double global_y = local_x * cos(car_yaw) + local_y * sin(car_yaw);

    return {global_x, global_y};
}

int ClosestWaypoint(double x, double y, const vector<double> &maps_x, const vector<double> &maps_y)
{

	double closestLen = 100000; //large number
	int closestWaypoint = 0;

	for(int i = 0; i < maps_x.size(); i++)
	{
		double map_x = maps_x[i];
		double map_y = maps_y[i];
		double dist = distance(x,y,map_x,map_y);
		if(dist < closestLen)
		{
			closestLen = dist;
			closestWaypoint = i;
		}

	}

	return closestWaypoint;

}

int NextWaypoint(double x, double y, double theta, const vector<double> &maps_x, const vector<double> &maps_y)
{

	int closestWaypoint = ClosestWaypoint(x,y,maps_x,maps_y);

	double map_x = maps_x[closestWaypoint];
	double map_y = maps_y[closestWaypoint];

	double heading = atan2((map_y-y),(map_x-x));

	double angle = fabs(theta-heading);
	angle = min(2*pi() - angle, angle);

	if(angle > pi()/4)
	{
		closestWaypoint++;
		if (closestWaypoint == maps_x.size())
		{
			closestWaypoint = 0;
		}
	}

	return closestWaypoint;
}

// Transform from Cartesian x,y coordinates to Frenet s,d coordinates
vector<double> getFrenet(double x, double y, double theta, const vector<double> &maps_x, const vector<double> &maps_y)
{
	int next_wp = NextWaypoint(x,y, theta, maps_x,maps_y);

	int prev_wp;
	prev_wp = next_wp-1;
	if(next_wp == 0)
	{
		prev_wp  = maps_x.size()-1;
	}

	double n_x = maps_x[next_wp]-maps_x[prev_wp];
	double n_y = maps_y[next_wp]-maps_y[prev_wp];
	double x_x = x - maps_x[prev_wp];
	double x_y = y - maps_y[prev_wp];

	// find the projection of x onto n
	double proj_norm = (x_x*n_x+x_y*n_y)/(n_x*n_x+n_y*n_y);
	double proj_x = proj_norm*n_x;
	double proj_y = proj_norm*n_y;

	double frenet_d = distance(x_x,x_y,proj_x,proj_y);

	//see if d value is positive or negative by comparing it to a center point

	double center_x = 1000-maps_x[prev_wp];
	double center_y = 2000-maps_y[prev_wp];
	double centerToPos = distance(center_x,center_y,x_x,x_y);
	double centerToRef = distance(center_x,center_y,proj_x,proj_y);

	if(centerToPos <= centerToRef)
	{
		frenet_d *= -1;
	}

	// calculate s value
	double frenet_s = 0;
	for(int i = 0; i < prev_wp; i++)
	{
		frenet_s += distance(maps_x[i],maps_y[i],maps_x[i+1],maps_y[i+1]);
	}

	frenet_s += distance(0,0,proj_x,proj_y);

	return {frenet_s,frenet_d};

}

// Transform from Frenet s,d coordinates to Cartesian x,y
vector<double> getXY(double s, double d, const vector<double> &maps_s, const vector<double> &maps_x, const vector<double> &maps_y)
{
	int prev_wp = -1;

	while(s > maps_s[prev_wp+1] && (prev_wp < (int)(maps_s.size()-1) ))
	{
		prev_wp++;
	}

	int wp2 = (prev_wp+1)%maps_x.size();

	double heading = atan2((maps_y[wp2]-maps_y[prev_wp]),(maps_x[wp2]-maps_x[prev_wp]));
	// the x,y,s along the segment
	double seg_s = (s-maps_s[prev_wp]);

	double seg_x = maps_x[prev_wp]+seg_s*cos(heading);
	double seg_y = maps_y[prev_wp]+seg_s*sin(heading);

	double perp_heading = heading-pi()/2;

	double x = seg_x + d*cos(perp_heading);
	double y = seg_y + d*sin(perp_heading);
	return {x,y};

}

//vector<vector<double>, vector<double>> sortXYPoints (vector<double> xpoints, vector<double> ypoints) {
//    vector<int> index(xpoints.size(), 0);
//    for (int i = 0 ; i != index.size() ; i++) {
//        index[i] = i;
//    }
//    sort(index.begin(), index.end(),
//         [&](const int& a, const int& b) {
//             return (xpoints[a] < xpoints[b]);
//         }
//    );
//    vector<double> yout;
//    for (int i = 0 ; i != index.size() ; i++) {
//        yout.push_back(ypoints[index[i]]);
//    }
//    return {xpoints, yout};
//};

int main() {
	uWS::Hub h;

	// Load up map values for waypoint's x,y,s and d normalized normal vectors
	vector<double> map_waypoints_x;
	vector<double> map_waypoints_y;
	vector<double> map_waypoints_s;
	vector<double> map_waypoints_dx;
	vector<double> map_waypoints_dy;

	// Waypoint map to read from
	string map_file_ = "../data/highway_map.csv";
	// The max s value before wrapping around the track back to 0
	double max_s = 6945.554;

	ifstream in_map_(map_file_.c_str(), ifstream::in);

	string line;
	while (getline(in_map_, line)) {
		istringstream iss(line);
		double x;
		double y;
		float s;
		float d_x;
		float d_y;
		iss >> x;
		iss >> y;
		iss >> s;
		iss >> d_x;
		iss >> d_y;
		map_waypoints_x.push_back(x);
		map_waypoints_y.push_back(y);
		map_waypoints_s.push_back(s);
		map_waypoints_dx.push_back(d_x);
		map_waypoints_dy.push_back(d_y);
	}

	h.onMessage([&map_waypoints_x,&map_waypoints_y,&map_waypoints_s,&map_waypoints_dx,&map_waypoints_dy](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
																										 uWS::OpCode opCode) {
		// "42" at the start of the message means there's a websocket message event.
		// The 4 signifies a websocket message
		// The 2 signifies a websocket event
		//auto sdata = string(data).substr(0, length);
		//cout << sdata << endl;
		if (length && length > 2 && data[0] == '4' && data[1] == '2') {

			auto s = hasData(data);

			if (s != "") {
				auto j = json::parse(s);

				string event = j[0].get<string>();

				if (event == "telemetry") {
					// j[1] is the data JSON object

					// Main car's localization Data
					double car_x = j[1]["x"];
					double car_y = j[1]["y"];
					double car_s = j[1]["s"];
					double car_d = j[1]["d"];
					double car_yaw = j[1]["yaw"];
					double car_speed = j[1]["speed"];

					// Previous path data given to the Planner
					auto previous_path_x = j[1]["previous_path_x"];
					auto previous_path_y = j[1]["previous_path_y"];
					// Previous path's end s and d values
					double end_path_s = j[1]["end_path_s"];
					double end_path_d = j[1]["end_path_d"];

					// Sensor Fusion Data, a list of all other cars on the same side of the road.
					auto sensor_fusion = j[1]["sensor_fusion"];

					int prevpath_size = previous_path_x.size();

					json msgJson;
                    //points for plotting next path
					vector<double> next_x_vals;
					vector<double> next_y_vals;

					//points for fitting spline
					vector<double> points_x;
					vector<double> points_y;

					const int points_increment = 15;

					//where the car is
                    double ref_x = car_x;
                    double ref_y = car_y;
                    double ref_yaw = deg2rad(car_yaw);
                    double ref_s = car_s;
                    double ref_d = car_d;
//                    double ref_yaw = deg2rad(car_yaw);

                    if (prevpath_size < 2) {
                        double prev_car_x = car_x - cos(ref_yaw);
                        double prev_car_y = car_x - sin(ref_yaw);

                        points_x.push_back(prev_car_x);
                        points_y.push_back(prev_car_y);

                        points_x.push_back(car_x);
                        points_y.push_back(car_y);

                        cout<< "here"<< prev_car_x<< ","<< car_x;


                    } else {
                        ref_x = previous_path_x[prevpath_size - 1];
                        ref_y = previous_path_y[prevpath_size - 1];
                        ref_s = getFrenet(ref_x, ref_y, ref_yaw, map_waypoints_x, map_waypoints_y).at(0);
                        ref_d = getFrenet(ref_x, ref_y, ref_yaw, map_waypoints_x, map_waypoints_y).at(1);

                        double ref_x_prev = previous_path_x[prevpath_size - 2];
                        double ref_y_prev = previous_path_y[prevpath_size - 2];

                        ref_yaw = atan2((ref_y - ref_y_prev), (ref_x - ref_x_prev));
                        if (fabs(ref_yaw) > deg2rad(25)) {
                            ref_yaw = 0;
                        }

                        points_x.push_back(ref_x_prev);
                        points_x.push_back(ref_x);

                        points_y.push_back(ref_y_prev);
                        points_y.push_back(ref_y);

//                        points_x.push_back(car_x);
//                        points_y.push_back(car_y);

                    }
                    vector<double> next_wp0 = getXY((ref_s + 30), (2 + 4 * lane), map_waypoints_s, map_waypoints_x ,map_waypoints_y);
                    vector<double> next_wp1 = getXY((ref_s + 60), (2 + 4 * lane), map_waypoints_s, map_waypoints_x ,map_waypoints_y);
                    vector<double> next_wp2 = getXY((ref_s + 90), (2 + 4 * lane), map_waypoints_s, map_waypoints_x ,map_waypoints_y);

                    points_x.push_back(next_wp0[0]);
                    points_x.push_back(next_wp1[0]);
                    points_x.push_back(next_wp2[0]);

                    points_y.push_back(next_wp0[1]);
                    points_y.push_back(next_wp1[1]);
                    points_y.push_back(next_wp2[1]);

//                    for (int i =1; i < 5; i++) {
//                        double next_s = car_s + (i + 1) * points_increment;
//                        double next_d = 2 + 4 * lane;
//                        vector<double> next_xy = getXY(next_s, next_d, map_waypoints_s, map_waypoints_x ,map_waypoints_y);
//                        points_x.push_back(next_xy.at(0));
//                        points_y.push_back(next_xy.at(1));
//
//                        cout << "nextx"<< next_xy.at(0)<<","<< next_xy.at(1)<< endl;
//                    }



//                    for (int i = 0; i< points_x.size(); i++) {
//                        cout<< points_x.at(i)<< endl;
//                    }

                    for (int i =0; i< points_x.size(); i++) {
                        cout<< "point_x_testing"<< endl;
                        cout << "pointsx"<< points_x.at(i)<<","<< points_y.at(i)<< endl;
                        cout << "ref_x"<<ref_x<< endl;
                        cout << "ref_yaw"<<ref_yaw<< endl;
                        //shift the coordinates
                        double shifted_x = points_x.at(i) - ref_x;
                        double shifted_y = points_y.at(i) - ref_y;

                        //then rotate
                        points_x[i] = (shifted_x * cos(0 - ref_yaw) - shifted_y * sin(0 -ref_yaw));
                        points_y[i] = (shifted_x * sin(0 - ref_yaw) + shifted_y * cos(0 -ref_yaw));
                        cout << "pointsx_shifted"<< points_x.at(i)<<","<< points_y.at(i)<< endl;
                    }

                    tk::spline s;
//                    vector<vector<double>, vector<double>> sortedXY = sortXYPoints(points_x, points_y);
                    //anchor points for spline

//                    for (int i = 0; i< sortedXY.size(); i++) {
//                        cout<< sortedXY.at(i)<< endl;
//                    }
//                    s.set_points(sortedXY[0], sortedXY[1]);
                    s.set_points(points_x, points_y);

                    //add in prev path points
                    for (int i = 0; i < previous_path_x.size(); i++) {
                        next_x_vals.push_back(previous_path_x.at(i));
                        next_y_vals.push_back(previous_path_y.at(i));
                    }

                    //N * 0.02 * v = d where N == number of points, 0.02 is the rate at which car visits points, v = speed of car,
                    // d == distance in a straight line between car starting point and end point of the path
                    double horizon_x = 30;
                    double horizon_y = s(horizon_x);
                    double target_d =  sqrt(horizon_x *  horizon_x + horizon_y * horizon_y);
                    double N = target_d / ( max_speed * 0.02 / 2.24);
                    double x_sum_prev = 0; // the distance in x covered by previous points

                    for (int i = 1; i <= 50 - previous_path_x.size(); i++) {
                        double x_point_curr = x_sum_prev + horizon_x / N;
                        double y_point_curr = s(x_point_curr);

                        x_sum_prev = x_point_curr;

                        double x_ref = x_point_curr;
                        double y_ref = y_point_curr;

                        x_point_curr = (x_ref * cos(ref_yaw) - y_ref * sin(ref_yaw));
                        y_point_curr = (x_ref * sin(ref_yaw) + y_ref * cos(ref_yaw));

                        vector<double> gcoords = convert2GlobalCoords({x_point_curr, y_point_curr}, ref_yaw);

                        next_x_vals.push_back(x_point_curr + x_ref);
                        next_y_vals.push_back(y_point_curr + y_ref);
                    }
					msgJson["next_x"] = next_x_vals;
					msgJson["next_y"] = next_y_vals;

					auto msg = "42[\"control\","+ msgJson.dump()+"]";

					//this_thread::sleep_for(chrono::milliseconds(1000));
					ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);

				}
			} else {
				// Manual driving
				std::string msg = "42[\"manual\",{}]";
				ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
			}
		}
	});

	// We don't need this since we're not using HTTP but if it's removed the
	// program
	// doesn't compile :-(
	h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
					   size_t, size_t) {
		const std::string s = "<h1>Hello world!</h1>";
		if (req.getUrl().valueLength == 1) {
			res->end(s.data(), s.length());
		} else {
			// i guess this should be done more gracefully?
			res->end(nullptr, 0);
		}
	});

	h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
		std::cout << "Connected!!!" << std::endl;
	});

	h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
						   char *message, size_t length) {
		ws.close();
		std::cout << "Disconnected" << std::endl;
	});

	int port = 4567;
	if (h.listen(port)) {
		std::cout << "Listening to port " << port << std::endl;
	} else {
		std::cerr << "Failed to listen to port" << std::endl;
		return -1;
	}
	h.run();
}
