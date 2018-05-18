//
// Created by Wing Yee mak on 15/04/2018.
//
#include <math.h>

#include <chrono>
#include <iostream>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "utils.h"

using namespace Eigen;

constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Evaluate a polynomial.
double polyeval(Eigen::VectorXd coeffs, double x) {
    double result = 0.0;
    for (size_t i = 0; i < coeffs.size(); i++) {
        result += coeffs[i] * pow(x, i);
    }
    return result;
}

// get the differential value of a polynomial at position x
double polyDiffeval(Eigen::VectorXd coeffs, double x, int polydegree) {
    double result = 0.0;
    int mult = 1;
    for (int j = polydegree; j > 0; j--) {
        mult *= j;
    }
    for (int i = polydegree; i <coeffs.size() -1; i++) {
        result += coeffs[i] * mult * pow(x, (i-1));
    }
    return result;
}
// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals,
                        int order) {
    assert(xvals.size() == yvals.size());
    assert(order >= 1 && order <= xvals.size() - 1);
    Eigen::MatrixXd A(xvals.size(), order + 1);

    for (size_t i = 0; i < xvals.size(); i++) {
        A(i, 0) = 1.0;
    }

    for (size_t j = 0; j < xvals.size(); j++) {
        for (int i = 0; i < order; i++) {
            A(j, i + 1) = A(j, i) * xvals(j);
        }
    }

    auto Q = A.householderQr();
    auto result = Q.solve(yvals);
    return result;
}

// calculate radius of curvature at position x of the equation defined by coeffs
double radiusOfCurvature(Eigen::VectorXd coeffs, double x){
    double diff2 = polyDiffeval(coeffs, x, 2);
    double diff1 = polyDiffeval(coeffs, x, 1);

    return pow((1 + pow(1/diff1, 2)),1.5)/pow(diff2, 2);
};

//convert mph (from the simulator) to meters per second (for the calcuations)
double mileph2meterps(double miles) {return miles * 0.44704;}
//conversions the otherway
double meterps2mileph(double meters) {return meters / 0.44704;}
