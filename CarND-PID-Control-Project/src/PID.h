#ifndef PID_H
#define PID_H

#include <vector>
#include <tuple>

using namespace std;

class PID {
public:
    /*
    * Errors
    */
    double p_error;
    double i_error;
    double d_error;

    /*
    * Coefficients
    */
    double Kp;
    double Ki;
    double Kd;

    double curr_timestep;

    bool is_optimised = false;

    double twiddle_err_threshold;
    int curr_twiddle_iter;
    double best_error;
    vector<double> twiddle_dp;

    /*
    * Constructor
    */
    PID();

    /*
    * Destructor.
    */
    virtual ~PID();

    /*
    * Initialize PID.
    */
    void Init(double Kp, double Ki, double Kd);

    /*
    * Update the PID error variables given cross track error.
    */
    void UpdateError(double cte);

    /*
    * Calculate the total PID error.
    */
    double TotalError();

    void TwiddleUpate();

    vector<double> Twiddle(float tolerance);

//    tuple<vector, vector, double> Iter(vector params, int steps);
};

#endif /* PID_H */

