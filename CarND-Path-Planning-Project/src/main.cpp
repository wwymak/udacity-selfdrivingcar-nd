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
#include "util.h"
#include "spline.h"

using namespace std;

// for convenience
using json = nlohmann::json;


const int NUM_PRED_WAYPOINTS = 50;
const double MAX_SPEED = 49;
int lane = 1;
double ref_speed = 0;

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

                    json msgJson;

                    int prevpath_size = previous_path_x.size();
                    double horizon_x = 30.0;
                    bool too_close = false;


                    vector<double> next_x_vals;
                    vector<double> next_y_vals;
                    vector<double> pts_x;
                    vector<double> pts_y;

                    double ref_x = car_x;
                    double ref_y = car_y;
                    double ref_yaw = deg2rad(car_yaw);
                    double ref_s = car_s;

                    double ref_x_prev = car_x - cos(ref_yaw);
                    double ref_y_prev = car_y - sin(ref_yaw);


                    if(ref_yaw > deg2rad(25)) {
                        ref_yaw = deg2rad(25);
                    }

                    for (int i = 0; i < sensor_fusion.size(); i++) {
                        float d = sensor_fusion.at(i).at(6);
                        // if other car is in our lane:
                        if (d > (2 + 4 * lane -2) && d < (2 + 4 * lane + 2)) {
                            double vx = sensor_fusion.at(i).at(3);
                            double vy = sensor_fusion.at(i).at(4);
                            double other_speed = sqrt(vx * vx + vy * vy);
                            double other_car_s = sensor_fusion.at(i).at(5);

                            other_car_s += ((double)prevpath_size * 0.02 * other_speed);
                            if((other_car_s > ref_s) && ((other_car_s - ref_s) < horizon_x)) {
//                                ref_speed = other_speed;
                                too_close = true;
                                cout << "too close!!"<< endl;
                            }
                        }
                    }

                    if (too_close == true || ref_speed > MAX_SPEED) {
                        ref_speed -= 0.2;
                    } else if(ref_speed < MAX_SPEED - 0.2) {
                        ref_speed += 0.2;
                    }

                    if (prevpath_size >=  2) {
                       ref_x = previous_path_x[prevpath_size -1];
                       ref_y = previous_path_y[prevpath_size -1];

                       ref_x_prev = previous_path_x[prevpath_size -2];
                       ref_y_prev = previous_path_y[prevpath_size -2];

                       ref_yaw = atan2((ref_y - ref_y_prev), (ref_x - ref_x_prev));
                       ref_s = end_path_s;

                   }
                    pts_x.push_back(ref_x_prev);
                    pts_x.push_back(ref_x);

                    pts_y.push_back(ref_y_prev);
                    pts_y.push_back(ref_y);

                    for (int i = 1; i < 4; i++) {
                        vector<double> next_wp= getXY((ref_s + (i * 30)), (2 + 4 * lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
                        if (next_wp[0] > pts_x.at(1)){
                            pts_x.push_back(next_wp[0]);
                            pts_y.push_back(next_wp[1]);
                        }
                    }
                    vector<double> fitpts_x;
                    vector<double> fitpts_y;
                    for (int i = 0; i< pts_x.size(); i++) {
                        double shifted_x = pts_x.at(i) - ref_x;
                        double shifted_y = pts_y.at(i) - ref_y;
//                        cout << i << "ref:" << ref_x << ","<< ref_y<< "x, y:"<<pts_x.at(i)<< ","<< pts_y.at(i)<<  endl;
//                        cout << "shifted"<< shifted_x << "," << shifted_y<< endl;
                        double converted_x = shifted_x * cos(0 - ref_yaw) - shifted_y * sin(0 -ref_yaw);
                        double converted_y = shifted_x * sin(0 - ref_yaw) + shifted_y * cos(0 -ref_yaw);
                        if (i == 0 || converted_x > fitpts_x.at(i -1)){
                            fitpts_x.push_back(converted_x);
                            fitpts_y.push_back(converted_y);
                        }
//                        pts_x.at(i) = shifted_x * cos(0 - ref_yaw) - shifted_y * sin(0 -ref_yaw);
//                        pts_y.at(i) = shifted_x * sin(0 - ref_yaw) + shifted_y * cos(0 -ref_yaw);
//                        cout << i<< ":"<<"(" << pts_x.at(i)<< "," << pts_y.at(i)<< ")" << " ref yaw "<< rad2deg(ref_yaw)<< endl;
                    }

                    tk::spline s;
                    s.set_points(fitpts_x, fitpts_y);
                    for (int i = 0; i < prevpath_size; i++) {
                        next_x_vals.push_back(previous_path_x[i]);
                        next_y_vals.push_back(previous_path_y[i]);
                    }



                    double horizon_y = s(horizon_x);

                    double target_d =  sqrt(horizon_x *  horizon_x + horizon_y * horizon_y);
//                    double N = target_d / ( mileph2meterps(MAX_SPEED) * 0.02);
                    double x_sum_prev = 0; // the distance in x covered by previous points

                    for (int i = 0; i< 50 - prevpath_size; i++) {
                        double N = target_d / ( ref_speed * 0.02 / 2.24);
                        double x_point = x_sum_prev + horizon_x / N;
                        double y_point = s(x_point);

                        double x_ref = x_point;
                        double y_ref = y_point;

                        x_sum_prev = x_point;

                        x_point = (x_ref * cos(ref_yaw) - y_ref * sin(ref_yaw));
                        y_point = (x_ref * sin(ref_yaw) + y_ref * cos(ref_yaw));
//
//                        x_point += ref_x;
//                        y_point += ref_y;

                        next_x_vals.push_back(x_point + ref_x);
                        next_y_vals.push_back(y_point + ref_y);
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