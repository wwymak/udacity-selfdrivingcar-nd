//
// Created by Wing Yee mak on 19/05/2018.
//
#include <math.h>
#include <chrono>
#include <iostream>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include <vector>

using namespace Eigen;
using namespace std;

#ifndef PATH_PLANNING_UTIL_H
#define PATH_PLANNING_UTIL_H


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

double distance(double x1, double y1, double x2, double y2);
int ClosestWaypoint(double x, double y, const vector<double> &maps_x, const vector<double> &maps_y);

int NextWaypoint(double x, double y, double theta, const vector<double> &maps_x, const vector<double> &maps_y);

// Transform from Cartesian x,y coordinates to Frenet s,d coordinates
vector<double> getFrenet(double x, double y, double theta, const vector<double> &maps_x, const vector<double> &maps_y);

// Transform from Frenet s,d coordinates to Cartesian x,y
vector<double> getXY(double s, double d, const vector<double> &maps_s, const vector<double> &maps_x, const vector<double> &maps_y);

#endif //PATH_PLANNING_UTIL_H
