#include <uWS/uWS.h>
#include <iostream>
#include <fstream>
#include <string>
#include "json.hpp"
#include "PID.h"
#include <math.h>

// for convenience
using json = nlohmann::json;
using namespace std;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

//double Kp = 0.5;
//double Ki = 0;
//double Kd = 0.2;
//okay coeffs p:0.25 i:0 d:0.9
//okay coeffs p:0.10 i:0 d:1.0

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
std::string hasData(std::string s) {
    auto found_null = s.find("null");
    auto b1 = s.find_first_of("[");
    auto b2 = s.find_last_of("]");
    if (found_null != std::string::npos) {
        return "";
    }
    else if (b1 != std::string::npos && b2 != std::string::npos) {
        return s.substr(b1, b2 - b1 + 1);
    }
    return "";
}

int main(int argc, char *argv[])
{
    uWS::Hub h;

    PID pid;
    PID pid_v;

    double Kp = atof(argv[1]);
    double Ki = atof(argv[2]);
    double Kd = atof(argv[3]);

    double Kp_v = atof(argv[4]);
    double Ki_v = atof(argv[5]);
    double Kd_v = atof(argv[6]);

    double target_speed = 60.0;
    double min_speed = 30.0;

    ofstream logfile;
    auto filename = "./../tuning_logs/pid-p=" + to_string(Kp).substr (0,3) + "Ki=" + to_string(Ki).substr (0,3) + "Kd=" + to_string(Kd).substr (0,3)
                    + "pidv-p=" + to_string(Kp_v).substr (0,3) + "Ki=" + to_string(Ki_v).substr (0,3) + "Kd=" + to_string(Kd_v).substr (0,3);
    logfile.open(filename);
    logfile << "cte,steer,angle\n";


    pid.Init(Kp, Ki, Kd);
    pid_v.Init(Kp_v, Ki_v, Kd_v);

    h.onMessage([&pid, &pid_v, &target_speed, &logfile](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, uWS::OpCode opCode) {
        // "42" at the start of the message means there's a websocket message event.
        // The 4 signifies a websocket message
        // The 2 signifies a websocket event
        if (length && length > 2 && data[0] == '4' && data[1] == '2')
        {
            auto s = hasData(std::string(data).substr(0, length));
            if (s != "") {
                auto j = json::parse(s);
                std::string event = j[0].get<std::string>();
                if (event == "telemetry") {
                    // j[1] is the data JSON object
                    double cte = std::stod(j[1]["cte"].get<std::string>());
                    double speed = std::stod(j[1]["speed"].get<std::string>());
                    double angle = std::stod(j[1]["steering_angle"].get<std::string>());
                    double steer_value;
                    double throttle_val;
                    /*
                    * TODO: Calcuate steering value here, remember the steering value is
                    * [-1, 1].
                    * NOTE: Feel free to play around with the throttle and speed. Maybe use
                    * another PID controller to control the speed!
                    */
                    cout << angle<< "angle"<< endl;
                    pid.UpdateError(cte);
                    pid_v.UpdateError(speed - target_speed);
                    steer_value = -pid.TotalError();
                    throttle_val = -pid_v.TotalError();
                    //stop accelerating now as the car is moving out of the track
                    if (fabs(cte) > 1.5) {
                        throttle_val = 0;
                    }
                    // car oscillating far too much, have to brake
                    if (fabs(cte) > 5.0) {
                        throttle_val = -1.0;
                    }


                    if(steer_value > 1) {
                        steer_value = -1;
                    } else if (steer_value < -1) {
                        steer_value = 1;
                    }
                    if(throttle_val > 1) {
                        throttle_val = 1;
                    } else if (throttle_val < -1) {
                        throttle_val = -1;
                    }
//                    steer_value = angle + deg2rad(- pid.Kp * pid.d_error - pid.Ki * pid.i_error - pid.Kd * pid.d_error);
                    // DEBUG
                    std::cout << "CTE: " << cte << " Steering Value: " << steer_value << std::endl;
                    logfile<<cte<<","<< steer_value <<"," << angle << "\n" ;
                    json msgJson;
                    msgJson["steering_angle"] = steer_value;
                    msgJson["throttle"] = throttle_val;
//                    msgJson["throttle"] = 0.3;
                    auto msg = "42[\"steer\"," + msgJson.dump() + "]";
                    std::cout << msg << std::endl;
                    ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);


                }
            } else {
                // Manual driving
                std::string msg = "42[\"manual\",{}]";
                ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
            }
        }
    });

    // We don't need this since we're not using HTTP but if it's removed the program
    // doesn't compile :-(
    h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data, size_t, size_t) {
        const std::string s = "<h1>Hello world!</h1>";
        if (req.getUrl().valueLength == 1)
        {
            res->end(s.data(), s.length());
        }
        else
        {
            // i guess this should be done more gracefully?
            res->end(nullptr, 0);
        }
    });

    h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
        std::cout << "Connected!!!" << std::endl;
    });

    h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code, char *message, size_t length) {
        ws.close();
        std::cout << "Disconnected" << std::endl;
    });

    int port = 4567;
    if (h.listen(port))
    {
        std::cout << "Listening to port " << port << std::endl;
    }
    else
    {
        std::cerr << "Failed to listen to port" << std::endl;
        return -1;
    }
    h.run();
}
