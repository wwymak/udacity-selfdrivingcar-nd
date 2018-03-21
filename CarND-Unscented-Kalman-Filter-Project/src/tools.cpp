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
        residual = estimations[i] - ground_truth[i];
        residual = residual.array() * residual.array();
        rmse += residual;
    }


    rmse = rmse / n;
    rmse = rmse.array().sqrt();
    return rmse;
}

/**
 * convert the angle phi from state vector to between pi and -pi
 * @param phi state angle to normalises
 */
void Tools::NormaliseAngles(double &phi) {
    phi = atan2(sin(phi), cos(phi));
}