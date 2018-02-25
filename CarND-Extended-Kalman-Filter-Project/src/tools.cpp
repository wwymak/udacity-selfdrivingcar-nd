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
  /**
  TODO:
    * Calculate a Jacobian here.
  */
    MatrixXd Hj(3,4);
    Hj.setZero();




}
