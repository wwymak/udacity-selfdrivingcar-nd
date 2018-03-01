#include "kalman_filter.h"
#include "tools.h"
#include <cmath>
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

Tools tools;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
  * prediction for filter
  */
    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
    * normal Kalman Filter equations for state update
  */

    long xsize = x_.size();
    //identity matrix
    MatrixXd I = MatrixXd::Identity(xsize, xsize);

    VectorXd y = z - H_ * x_;
    MatrixXd S = H_ * P_ * H_.transpose();
    //Kalman gain
    MatrixXd K = P_ * H_.transpose();

    x_ = x_ + K * y;
    P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */
    VectorXd h(3);

    const double pi = acos(-1);

    float px = x_(0);
    float py = x_(1);
    float vx = x_(2);
    float vy = x_(3);

    float p_mag2 = pow(px, 2) + pow(py, 2);
    float p_mag = pow(p_mag2, 0.5);

    h(0) = p_mag;
    h(1) = atan2(py , px);
    h(2) = (px * vx + py * vy)/p_mag;

    long xsize = x_.size();
    //identity matrix
    MatrixXd I = MatrixXd::Identity(xsize, xsize);

    VectorXd y = z - h;
    while (y(1) > pi) {
        y(1) -= 2 * pi;
    }
    while (y(1) < -pi) {
        y(1) += 2 * pi;
    }

    MatrixXd S = H_ * P_ * H_.transpose();
    //Kalman gain
    MatrixXd K = P_ * H_.transpose();

    x_ = x_ + K * y;
    P_ = (I - K * H_) * P_;


}
