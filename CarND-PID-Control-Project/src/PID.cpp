#include "PID.h"
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
    this->Kp = Kp;
    this->Ki = Ki;
    this->Kd = Kd;

    d_error = 0;
    i_error = 0;
    p_error = 0;

    prev_input=1000;
    diff_input = 0;
}

void PID::UpdateError(double cte, double steer_angle) {
    if(abs(prev_input) > 25) {
        prev_input = steer_angle;
    }
    i_error += cte;
    d_error = cte - p_error;
    p_error = cte;

    diff_input = steer_angle - prev_input;
    prev_input = steer_angle;
}

double PID::TotalError() {
//    return Kp * p_error + Ki * i_error + Kd * d_error - (Kd * diff_input)/180.0;
    return Kp * p_error + Ki * i_error + Kd * d_error;
}

