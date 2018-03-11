#include "kalman_filter.h"
#include "tools.h"
#include <cmath>
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;


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
    //position vector x
    long xsize = x_.size();
    //identity matrix
    MatrixXd I = MatrixXd::Identity(xsize, xsize);

    VectorXd y = z - H_ * x_;
    MatrixXd S = H_ * P_ * H_.transpose() + R_;
    //Kalman gain
    MatrixXd K = P_ * H_.transpose() * S.inverse();

    x_ = x_ + (K * y);
    P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
    /**
        * extended Kalman Filter equations for state update
      */
    VectorXd h(3);

    const double pi = acos(-1);

    float px = x_(0);
    float py = x_(1);
    float vx = x_(2);
    float vy = x_(3);
    //convert position and velociyt values to the polar coordinate space for radar
    float rho = sqrt(pow(px, 2) + pow(py, 2));
    float theta = atan2(py , px);
    float rho_dot = (px * vx + py * vy)/rho;

    float p_mag2 = pow(px, 2) + pow(py, 2);
    double p_mag = sqrt(p_mag2);

    h << rho, theta, rho_dot;
    long xsize = x_.size();
    //identity matrix
    MatrixXd I = MatrixXd::Identity(xsize, xsize);

    VectorXd y = z - h;
    //normalise the angles so it's between pi and -pi
//    y(1) = fmod(y(1), 2.0 * pi);

    /**
     * todo figure out why fmod is doing strange things
     while the lectures solutions seem to be using the following to normalise angles,
    it is a rather inefficient way of going about things esp if you end up with a large angle,
        (and it might also get stuck in and infinite loop if you make a mistake in your angle cacl)
     */
        while (y(1) > pi) {
            y(1) -= 2 * pi;
        }
        while (y(1) < -pi) {
            y(1) += 2 * pi;
        }

    MatrixXd S = H_ * P_ * H_.transpose() + R_;
    //Kalman gain
    MatrixXd K = P_ * H_.transpose() * S.inverse();

    x_ = x_ + K * y;
    P_ = (I - K * H_) * P_;

}
