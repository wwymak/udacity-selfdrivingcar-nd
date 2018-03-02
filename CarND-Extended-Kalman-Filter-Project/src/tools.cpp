#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

//RMS error of the kalman filter output
VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    // initialse mean squared error
    VectorXd rmse(4);
    rmse.setZero();
    VectorXd residual(4);

    int n = estimations.size();

    // check if the estimation vector is the right size:
    if( n != ground_truth.size() || n == 0 || ground_truth.size() == 0) {
        cout << "error-- estimation vector wrong size: " << estimations.size()<< endl;
        return rmse;
    }
    //calculated the sum fo the residuals squared
    for (int i = 0; i< n; i++) {
        residual = estimations[i] - ground_truth[i];
        residual = residual.array() * residual.array();
        rmse += residual;
    }


    rmse = rmse / n;
    rmse = rmse.array().sqrt();
    return rmse;

}

//Jacobian H matrix for non linear radar kalman algorithm
MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
    //initialise Hj the jacobian matrix

    MatrixXd Hj(3,4);
    //recover state parameters
    double px = x_state(0);
    double py = x_state(1);
    double vx = x_state(2);
    double vy = x_state(3);

    //pre-compute a set of terms to avoid repeated calculation
    double p_mag2 = pow(px, 2) + pow(py, 2);
    double p_mag = sqrt(p_mag2);
    double p_mag3 = (p_mag2*p_mag);

    //check division by zero
    if(fabs(p_mag2) < 0.0001){
        cout << "CalculateJacobian () - Error - Division by Zero" << endl;
        return Hj;
    }

    //compute the Jacobian matrix
    Hj << (px/p_mag), (py/p_mag), 0, 0,
            -(py/p_mag2), (px/p_mag2), 0, 0,
            py*(vx*py - vy*px)/p_mag3, px*(px*vy - py*vx)/p_mag3, px/p_mag, py/p_mag;

    return Hj;

}
