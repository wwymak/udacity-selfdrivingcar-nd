#include "PID.h"
#include <iostream>

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
}

void PID::UpdateError(double cte) {
    i_error += cte;
    d_error = cte - p_error;
    p_error = cte;
}

double PID::TotalError() {
    return Kp * p_error + Ki * i_error + Kd * d_error;
}

vector<double> PID::Twiddle(float tolerance){
    vector<double> params = {Kp, Ki, Kd};
    vector<double> dp = {1,1,1};
    double dpSum;
    for (auto& n : dp){
        dpSum += n;
    };
    while (dpSum > tolerance) {
        for (int i = 0; i< dp.size(); i++) {
            params.at(i) += dp.at(i);
            IterRun(params, 50);
        }
    }
};

void PID::IterRun(vector params, int steps) {
    vector<double> x_trajectory = [];
    vector<double> y_trajectory = [];
};

