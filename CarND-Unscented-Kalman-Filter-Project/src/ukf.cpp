#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include <tuple>
#include "tools.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

const double pi = acos(-1);

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
    std_a_ = 0.2;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 0.2;

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

    previous_timestamp_ = 0;
    //todo figure out what this is
    ///* time when the state is true, in us
    time_us_ = 0;

    Tools tools;

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
            double rho = meas_package.raw_measurements_(0);
            double theta = meas_package.raw_measurements_(1);
            double rho_dot = meas_package.raw_measurements_(2);

            x_ << rho * cos(theta), rho * sin(theta), rho_dot, rho, 0;
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
        previous_timestamp_ =  meas_package.timestamp_;
        // done initializing, no need to predict or update

        //vector for weights
        //weights
        weights_.fill(0.5 / (lambda_ + n_aug_));
        weights_(0) = lambda_ / (lambda_ + n_aug_);
        is_initialized_ = true;
        return;
    } else {
        double dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;

        //update timestamps
        previous_timestamp_ =  meas_package.timestamp_;

        Prediction(dt);

        if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_ == true) {
            UpdateRadar(meas_package);
        } else if(meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_ == true){
            UpdateLidar(meas_package);
        }
        
    }
}

MatrixXd UKF::GenerateAugmentedSigmaPoints() {
    VectorXd x_aug = VectorXd(7);
    MatrixXd P_aug = MatrixXd(7,7);
    //sigma points matrix
    MatrixXd Xsigma = MatrixXd(n_aug_, 2 * n_aug_ + 1);

    //augmented mean state
    x_aug.head(5) = x_;
    //means of noise is zero, so x_aug(nu_a) and x_aug(nu_yaw) = 0
    x_aug(5) = 0;
    x_aug(6) = 0;

    //augemtned covariance matrix:
    P_aug.setZero();
    P_aug.topLeftCorner(5,5) = P_;
    P_aug(5,5) = pow(std_a_, 2);
    P_aug(6,6) = pow(std_yawdd_ , 2);


    //sqrt of P_aug
    MatrixXd A = P_aug.llt().matrixL();

    Xsigma.col(0) = x_aug;
    for (int i = 0; i< n_aug_; i++) {
        Xsigma.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * A.col(i);
        Xsigma.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * A.col(i);
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

tuple<VectorXd, MatrixXd> UKF::PredictMeanAndCovariance(MatrixXd Xsigma_pred) {
    //predicted state
    VectorXd x = VectorXd(n_x_);
    x.setZero();
    //preicte covariance matrix
    MatrixXd P = MatrixXd(n_x_, n_x_);
    P.setZero();

    //predicte state
    for (int i = 0; i < 2* n_aug_ +1; i++) {
        x = x + weights_(i) * Xsigma_pred.col(i);
    }
    //predicted covariance matrix:
    for (int i = 0; i < 2* n_aug_ +1; i++) {
        VectorXd xdiff = Xsigma_pred.col(i) - x;
        tools.NormaliseAngles(xdiff(3));
//        xdiff(3) = fmod(xdiff(3), 2.0 * pi);

        P = P + weights_(i) * xdiff * xdiff.transpose();
    }
    return std::make_tuple(x, P);

}

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
    std::tuple<VectorXd, MatrixXd> predMeanCov = PredictMeanAndCovariance(Xsig_pred_);
    //predicted mean
    x_ = std::get<0>(predMeanCov);
    //predicte covariance
    P_ = std::get<1>(predMeanCov);

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {

    /**
     * the lidar is a linear upate so can use the normal kalman filter upate here
    */
    MatrixXd H_laser_ = MatrixXd(2,5);
    H_laser_ << 1, 0, 0, 0, 0,
               0, 1, 0, 0,0;
    MatrixXd R_laser_ = MatrixXd(2,2);

    R_laser_ << std_laspx_ * std_laspx_, 0,
            0, std_laspy_ * std_laspy_;

    long xsize = x_.size();
    //identity matrix
    MatrixXd I = MatrixXd::Identity(xsize, xsize);

    VectorXd y = meas_package.raw_measurements_ - H_laser_ * x_;
    MatrixXd S = H_laser_ * P_ * H_laser_.transpose() + R_laser_;
    //Kalman gain
    MatrixXd K = P_ * H_laser_.transpose() * S.inverse();

    x_ = x_ + (K * y);
    P_ = (I - K * H_laser_) * P_;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  update the state x_ and covariance matrx P_ according to measurements from the radar
   the radar is non linear so using the unscented kalman filter algorithm here

  */
    //   Predict measuremnet sigma points for radar (ie put previous predicte sigma points into measruement space

    //matrix for sigma points in measurement space

    //dimensions of measurement for radar
    int n_z = 3; //since radar has rho, theta and rho_dot as dimensions
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

    //mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);
    z_pred.setZero();

    //measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z, n_z);
    S.setZero();

    for (int i = 0; i< 2 * n_aug_ + 1;  i++) {
        VectorXd xk = Xsig_pred_.col(i);
        double px = xk(0);
        double py = xk(1);
        double v = xk(2);
        double yaw = xk(3);
        double yawdot = xk(4);

        double rho_meas;
        double  theta_meas;
        double rho_dot_meas;

        rho_meas = sqrt(pow(px, 2) + pow(py, 2));
        theta_meas =  atan2(py, px);
        rho_dot_meas =  (px * cos(yaw) * v + py * sin(yaw) * v) / rho_meas;

        Zsig.col(i) << rho_meas, theta_meas , rho_dot_meas;
    }

    //mean predicted mesurment:
    for (int i = 0; i< 2 * n_aug_ + 1;  i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }


    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        //normalise angles
        tools.NormaliseAngles(z_diff(1));

        S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    //noise covariance matrix
    MatrixXd R= MatrixXd(3,3);
    R << pow(std_radr_, 2), 0,0,
        0, pow(std_radphi_, 2),0,
        0,0, pow(std_radrd_, 2);

    S = S+R;

    //cross-correlation matrix
    MatrixXd Tc = MatrixXd(n_x_,n_z);
    Tc.setZero();
    for (int i = 0; i< 2* n_aug_ + 1; i++) {
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        VectorXd zdiff =  Zsig.col(i) - z_pred;

        tools.NormaliseAngles(zdiff(1));
        //angle normalization
//        zdiff(1) = fmod(zdiff(1), 2.0 * pi);
//        while (zdiff(1) > M_PI) {
//            zdiff(1) -= 2. * M_PI;
//        }
//        while (zdiff(1) < -M_PI) {
//            zdiff(1) += 2.* M_PI;
//        };
        Tc = Tc + weights_(i) * x_diff * zdiff.transpose();
    }
    //kalmain gain
    MatrixXd K = MatrixXd(n_x_, n_z);
    K = Tc * S.inverse();

    x_ = x_ + K * ( meas_package.raw_measurements_ - z_pred);
    P_ = P_ - K * S * K.transpose();
}
