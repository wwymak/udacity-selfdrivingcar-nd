#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    // initialse mean squared error
    VectorXd rmse(4);
    rmse.setZero();
    VectorXd residual(4);

    // check if the estimation vector is the right size:
    if(estimations.size() != ground_truth.size() || estimations.size() == 0) {
        cout << "error-- estimation vector wrong size: " << estimations.size()<< endl;
        return rmse;
    }
    //calculated the sum fo the residuals squared
    for (int i = 0; i< estimations.size(); i++) {
        residual = estimations[i] - ground_truth[i];
        residual = residual.array() * residual.array();
        rmse += residual;
    }

    rmse = rmse / estimations.size();
    rmse = rmse.array().sqrt();
    return rmse;

}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
    //initialise Hj the jacobian matrix
    MatrixXd Hj(3,4);
    Hj.setZero();

    //state parameters for calculating the componentws of Hj
    float px = x_state(0);
    float py = x_state(1);
    float vx = x_state(2);
    float vy = x_state(3);

    //calculate px^2 + py^2
    float p_mag2 = pow(px, 2) + pow(py, 2);
    float p_mag = pow(p_mag2, 0.5);

    if(p_mag < 0.00001) {
        cout << "division by zero in jacobian matrix calc: px=" << px << " py=" << py << endl;
        return Hj;
    }

    Hj(0,0) = px / p_mag;
    Hj(0,1) = py /p_mag;
    Hj(1,0) = -py /p_mag2;
    Hj(1,1) = px /p_mag2;
    Hj(2,0) = py * ( vx * py - vy * px) /pow(p_mag, 1.5);
    Hj(2,1) = px * ( vy * px - vx * py) /pow(p_mag, 1.5);
    Hj(2,2) = px / p_mag;
    Hj(2,3) = py / p_mag;

    return Hj;

}
