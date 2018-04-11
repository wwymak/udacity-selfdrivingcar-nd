#include "PID.h"
#include <iostream>

using namespace std;

/*
* PID class.
 * Nothing fancy here, just need to set the Kp, Ki and Kd terms,
 * implemennt the method to calculate the corrective term, and also a method
 * to keep track of the cumulated error and error at previous timestep to
 * calculate the Kd and Ki terms
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
}

void PID::UpdateError(double cte) {
    i_error += cte;
    d_error = cte - p_error;
    p_error = cte;
}

double PID::TotalError() {
    return Kp * p_error + Ki * i_error + Kd * d_error;
}

