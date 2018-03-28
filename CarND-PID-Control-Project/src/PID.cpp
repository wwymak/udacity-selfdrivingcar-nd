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
    twiddle_dp = {0.01,0.001,0.01};

    best_error = 99999;

}

void PID::UpdateError(double cte) {
    i_error += cte;
    d_error = cte - p_error;
    p_error = cte;

    if(best_error < twiddle_err_threshold) {

        is_optimised = true;
        cout << "optimsed:  " << best_error<<endl;
    } else if(fmod(curr_timestep,10) == 0) {
        TwiddleUpate();
        curr_twiddle_iter = (curr_twiddle_iter += 1) % 3;
    }
}

double PID::TotalError() {
    return Kp * p_error + Ki * i_error + Kd * d_error;
}

//vector<double> PID::Twiddle(float tolerance){
//    vector<double> params = {Kp, Ki, Kd};
//    vector<double> dp = {1,1,1};
//    double err = p_error;
//    double dpSum;
//    for (auto& n : dp){
//        dpSum += n;
//    };
//    while (dpSum > tolerance) {
//        for (int i = 0; i< dp.size(); i++) {
//            params.at(i) += dp.at(i);
//            Iter(params, 50);
//        }
//    }
//};

void PID::TwiddleUpate(){
//    double& paramToUpdate = Kp;
//    if(curr_twiddle_iter == 1) {
//        paramToUpdate = Ki;
//    } else if(curr_twiddle_iter == 2) {
//        paramToUpdate = Kd;
//    } else {
//        paramToUpdate = Kp;
//    }

    if(pow(p_error,2) < pow(best_error, 2)) {
        cout << best_error << ","<< p_error<< endl;
        best_error = abs(p_error);
        twiddle_dp.at(curr_twiddle_iter) *= 1.1;
        cout << "keep params" << twiddle_dp.at(0) <<","<<twiddle_dp.at(1) <<","<< twiddle_dp.at(2) <<","<< endl;

        cout << "keep params twiddle update: Kp"<< Kp << endl;
        cout << "keep params twiddle update: Ki"<< Ki << endl;
        cout << "keep params twiddle update: Kd"<< Kd << endl;
    } else {
        if(curr_twiddle_iter == 0) {
            Kp -= 2* twiddle_dp.at(curr_twiddle_iter);
            cout << "Kp change: Kp"<< Kp << endl;
            cout << "Kp change: Ki"<< Ki << endl;
            cout << "Kp change: Kd"<< Kd << endl;

        } else if(curr_twiddle_iter == 1) {
            Ki -= 2* twiddle_dp.at(curr_twiddle_iter);
            cout << "Ki change: Kp"<< Kp << endl;
            cout << "Ki change: Ki"<< Ki << endl;
            cout << "Ki change: Kd"<< Kd << endl;
        } else {
            Kd -= 2* twiddle_dp.at(curr_twiddle_iter);
            cout << "Kd change: Kp"<< Kp << endl;
            cout << "Kd change: Ki"<< Ki << endl;
            cout << "Kd change: Kd"<< Kd << endl;
        }
//        paramToUpdate -= 2* twiddle_dp.at(curr_twiddle_iter);
        twiddle_dp.at(curr_twiddle_iter) *= 0.9;
        cout << "twiddle change params" << twiddle_dp.at(0) <<","<<twiddle_dp.at(1) <<","<< twiddle_dp.at(2) <<","<< endl;


    }

    cout << "twiddle update: curr_twiddle_iter"<< curr_twiddle_iter << endl;
    cout << "twiddle update: Kp"<< Kp << endl;
    cout << "twiddle update: Ki"<< Ki << endl;
    cout << "twiddle update: Kd"<< Kd << endl;

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

