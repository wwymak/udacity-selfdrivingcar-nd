#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
    //state dimension
    n_x_ = 5;

    //Augmented state dimension
    n_aug_ = 7;

    ///* Sigma point spreading parameter
    lambda_= 3 - n_aug_;

    is_initialized_ = false;
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    // initial state vector
    x_ = VectorXd(5);

    // initial covariance matrix
    P_ = MatrixXd(5, 5);

    // Weights of sigma points
    weights_ = VectorXd(2 * n_aug_ +1);

    //predicted sigma points matrix
    Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 2;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 1;

    //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;
    //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.
    //todo figure out what this is
    ///* time when the state is true, in us
    time_us_ = 0;

}

UKF::~UKF() {

}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
    if (!is_initialized_) {
        //todo initialisationmkdi
        P_ = MatrixXd::Identity(5,5);

        if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
            /**
            Convert radar from polar to cartesian coordinates and initialize state.
            */
//            double rho = meas_package.raw_measurements_(0);
//            double theta = meas_package.raw_measurements_(1);
//            double rho_dot = meas_package.raw_measurements_(2);
//            x_(0) = rho * cos(theta);
//            x_(1) = rho * sin(theta);
//            x_(2) = rho_dot * cos(theta);
//            x_(3) = rho_dot * sin(theta);
        }
        else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
            /**
            Initialize state.
            */
            double px = meas_package.raw_measurements_(0);
            double py = meas_package.raw_measurements_(1);
            double v = 0.5;
            double yaw = 0.5;
            double yawdot = 0;

            x_ << px, py, v, yaw, yawdot;

        }
        time_us_ =  meas_package.timestamp_;
        // done initializing, no need to predict or update
        is_initialized_ = true;
        return;
    } else {
        double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
        double dt2 = pow(dt, 2);
        double dt3 = pow(dt, 3);
        double dt4 = pow(dt, 4);

        //update timestamps
        time_us_ =  meas_package.timestamp_;
        
        Prediction(dt);

        if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
            UpdateRadar(meas_package);
        } else {
            UpdateLidar(meas_package);
        }
        
    }
}

MatrixXd UKF::GenerateAugmentedSigmaPoints() {
    VectorXd x_aug = VectorXd(7);
    MatrixXd P_aug = MatrixXd(7,7);

    //augmented mean state
    x_aug.head(5) = x_;
    //means of noise is zero, so x_aug(nu_a) and x_aug(nu_yaw) = 0
    x_aug(5) = 0;
    x_aug(6) = 0;

    //augemtned covariance matrix:
    P_aug.setZero();
    P_aug.topLeftCorner(P_.size(), P_.size()) = P_;
    P_aug.bottomRightCorner(2,2) << std_a_ * std_a_ ,0,
                            0, std_yawdd_ * std_yawdd_;

    //sigma points matrix
    MatrixXd Xsigma = MatrixXd(n_aug_, 2 * n_aug_ + 1);
    //sqrt of P_aug
    MatrixXd A = P_aug.llt().matrixL();

    Xsigma.col(0) = x_aug;
    for (int i = 0; i< n_x_; i++) {
        Xsigma.col(i + 1) = x_aug + sqrt(lambda_ + n_x_) * A.col(i);
        Xsigma.col(i + 1 + n_x_) = x_aug - sqrt(lambda_ + n_x_) * A.col(i);
    }

    return Xsigma;

}

MatrixXd UKF::PredictSigmaPoints(double delta_t) {
    MatrixXd Xsigma_aug = GenerateAugmentedSigmaPoints();
    MatrixXd Xsigma_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);

    for (int i = 0; i< 2 * n_aug_ + 1; i++) {
        VectorXd xk = Xsigma_aug.col(i);
        double px = xk(0);
        double py = xk(1);
        double v = xk(2);
        double yaw = xk(3);
        double yawdot = xk(4);
        double nu_a = xk(5);
        double nu_yaw = xk(6);

        double px_pred;
        double py_pred;
        double v_pred;
        double  yaw_pred;
        double yaw_dot_pred;

        //handle cases when yawdot is zero
        if (fabs(yawdot) > 0.0001) {
            px_pred  = px + v/yawdot * (sin(yaw + yawdot * delta_t) - sin(yaw));
            py_pred = py + v/yawdot * (-cos(yaw + yawdot * delta_t) + cos(yaw));
        } else {
            px_pred  = v * cos(yaw) * delta_t;
            px_pred  = v * sin(yaw) * delta_t;
        }
        px_pred += 0.5 * pow(delta_t, 2) * cos(yaw) * nu_a;
        py_pred += 0.5 * pow(delta_t, 2) * sin(yaw) * nu_a;
        v_pred = v + delta_t * nu_a;
        yaw_pred = yaw + yawdot * delta_t + 0.5 * pow(delta_t, 2) * nu_yaw;
        yaw_dot_pred = yawdot + delta_t * nu_yaw;
        Xsigma_pred.col(i) << px_pred, py_pred , v_pred, yaw_pred, yaw_dot_pred;
    }

    return Xsigma_pred;
}

//void PredictMeanAndCovariance() {
//
//}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
    /**
    TODO:

    Complete this function! Estimate the object's location. Modify the state
    vector, x_. Predict sigma points, the state, and the state covariance matrix.
    */
    Xsig_pred_ = PredictSigmaPoints(delta_t);

    //create vector for weights
    weights_ = VectorXd(2 * n_aug_+1);


    //weights
    weights_.fill(0.5 / (lambda_ + n_aug_));
    weights_(0) = lambda_ / (lambda_ + n_aug_);

    //predicted mean
    for (int i = 0; i< 2 * n_aug_ +1; i++){
        x_ = x_ + weights_(i) * Xsig_pred_.col(i);
    }


    for (int i = 0; i< 2 * n_aug_ +1; i++){
        VectorXd xdiff = Xsig_pred_.col(i) - x_;
        P_ = P_ + weights_(i) * (Xsig_pred_.col(i) - x_) * (Xsig_pred_.col(i) - x_).transpose();
    }


}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
}
