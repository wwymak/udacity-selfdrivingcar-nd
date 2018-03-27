#include "PID.h"
#include <iostream>
#include <tuple>
#include <cmath>

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

    curr_timestep=0;
    curr_twiddle_iter = 0; //which param you are updating
    twiddle_err_threshold = 0.01;
    twiddle_dp = {1,1,1};

    best_error = 99999;

}

void PID::UpdateError(double cte) {
    i_error += cte;
    d_error = cte - p_error;
    p_error = cte;

    if(best_error < twiddle_err_threshold) {
        is_optimised = true;
    } else if(fmod(curr_timestep,10) == 0) {
        TwiddleUpate();
        curr_twiddle_iter = (curr_twiddle_iter += 1) % 3;
    }
}

double PID::TotalError() {
    return Kp * p_error + Ki * i_error + Kd * d_error;
}

vector<double> PID::Twiddle(float tolerance){
    vector<double> params = {Kp, Ki, Kd};
    vector<double> dp = {1,1,1};
    double err = p_error;
    double dpSum;
    for (auto& n : dp){
        dpSum += n;
    };
    while (dpSum > tolerance) {
        for (int i = 0; i< dp.size(); i++) {
            params.at(i) += dp.at(i);
            Iter(params, 50);
        }
    }
};

void PID::TwiddleUpate(){
    double& paramToUpdate = Kp;
    if(curr_twiddle_iter == 1) {
        paramToUpdate = Ki;
    } else if(curr_twiddle_iter == 2) {
        paramToUpdate = Kd;
    }

    if(p_error < best_error) {
        best_error = p_error;
        twiddle_dp.at(curr_twiddle_iter) *= 1.1;
    } else {
        paramToUpdate -= 2* twiddle_dp.at(curr_twiddle_iter);
        twiddle_dp.at(curr_twiddle_iter) *= 0.9;
    }

//    curr_twiddle_iter
}
//tuple<vector, vector, double> PID::Iter(vector params, int steps) {
//    vector<double> x_trajectory = [];
//    vector<double> y_trajectory = [];
//
//    double err = 0;
//
//    return make_tuple(x_trajectory, y_trajectory, err/steps);
//
//};

