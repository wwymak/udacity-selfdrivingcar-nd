//
// Created by Wing Yee mak on 15/04/2018.
//
#include <math.h>
#include <chrono>
#include <iostream>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"

using namespace Eigen;

#ifndef MPC_UTILS_H
#define MPC_UTILS_H
const double Lf = 2.67;

constexpr double pi();
double deg2rad(double x);
double rad2deg(double x);
double mileph2meterps(double miles);
double meterps2mileph(double meters);
double polyeval(Eigen::VectorXd coeffs, double x);
Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals,
                        int order);
double polyDiffeval(Eigen::VectorXd coeffs, double x);
double radiusOfCurvature(Eigen::VectorXd coeffs, double x);

#endif //MPC_UTILS_H
