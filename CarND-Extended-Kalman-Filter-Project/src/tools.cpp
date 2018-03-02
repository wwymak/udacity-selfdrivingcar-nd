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

    int n = estimations.size();

    // check if the estimation vector is the right size:
    if( n != ground_truth.size() || n == 0 || ground_truth.size() == 0) {
        cout << "error-- estimation vector wrong size: " << estimations.size()<< endl;
        return rmse;
    }
    //calculated the sum fo the residuals squared
    for (int i = 0; i< n; i++) {
        cout << "estimations"<<estimations[i] << endl;
        cout << "gt"<<ground_truth[i] << endl;
        residual = estimations[i] - ground_truth[i];
        residual = residual.array() * residual.array();
        rmse += residual;
    }


    rmse = rmse / n;
    rmse = rmse.array().sqrt();
    cout << "RMSE" << rmse<< endl;
    return rmse;

}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
    //initialise Hj the jacobian matrix
//    MatrixXd Hj(3,4);
//    Hj.setZero();
//
//    //state parameters for calculating the componentws of Hj
//    double px = x_state(0);
//    double py = x_state(1);
//    double vx = x_state(2);
//    double vy = x_state(3);
//
//    //calculate px^2 + py^2
//    double p_mag2 = pow(px, 2) + pow(py, 2);
//    double p_mag = pow(p_mag2, 0.5);
//
//    if(p_mag < 0.00001) {
//        cout << "division by zero in jacobian matrix calc: px=" << px << " py=" << py << endl;
//        return Hj;
//    }
//
//    Hj(0,0) = px / p_mag;
//    Hj(0,1) = py /p_mag;
//    Hj(1,0) = -py /p_mag2;
//    Hj(1,1) = px /p_mag2;
//    Hj(2,0) = py * ( vx * py - vy * px) /pow(p_mag, 1.5);
//    Hj(2,1) = px * ( vy * px - vx * py) /pow(p_mag, 1.5);
//    Hj(2,2) = px / p_mag;
//    Hj(2,3) = py / p_mag;
//
//    return Hj;

    MatrixXd Hj(3,4);
    //recover state parameters
    double px = x_state(0);
    double py = x_state(1);
    double vx = x_state(2);
    double vy = x_state(3);

    //pre-compute a set of terms to avoid repeated calculation
    double c1 = px*px+py*py;
    double c2 = sqrt(c1);
    double c3 = (c1*c2);

    //check division by zero
    if(fabs(c1) < 0.0001){
        cout << "CalculateJacobian () - Error - Division by Zero" << endl;
        return Hj;
    }

    //compute the Jacobian matrix
    Hj << (px/c2), (py/c2), 0, 0,
            -(py/c1), (px/c1), 0, 0,
            py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

    return Hj;

}
