#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"
#include "utils.h"

using CppAD::AD;

size_t N = 25;
//too small will be too ocmputational expensive
double dt = 0.05;

// This value assumes the model presented in the classroom is used.
//
// It was obtained by measuring the radius formed by running the vehicle in the
// simulator around in a circle with a constant steering angle and velocity on a
// flat terrain.
//
// Lf was tuned until the the radius formed by the simulating the model
// presented in the classroom matched the previous radius.
//
// This is the length from front to CoG that has a similar radius.
//const double Lf = 2.67;

//te target cte, steering angle err, target speed (although, the target speed changes depending on
//whether the car is going round a curve or not)
double ref_cte = 0;
double ref_epsi = 0;
double ref_v = 90;

size_t x_start = 0;
size_t y_start = x_start + N;
size_t psi_start = y_start + N;
size_t v_start = psi_start + N;
size_t cte_start = v_start + N;
size_t epsi_start = cte_start + N;
size_t delta_start = epsi_start + N;
size_t a_start = delta_start + N - 1;

double v_cost_weight = 1.0;
double cte_cost_weight = 5.0;
double epsi_cost_weight = 5.0;
double delta_cost_weight = 1.0;
double accel_cost_weight = 1.0;
double delta_change_cost_weight = 1.0;
double accel_change_cost_weight = 0.5;

class FG_eval {
public:
    // Fitted polynomial coefficients
    Eigen::VectorXd coeffs;
    double rCurve;
    FG_eval(Eigen::VectorXd coeffs, double rCurve) {
        this->coeffs = coeffs;
        this->rCurve = rCurve;
    }

    typedef CPPAD_TESTVECTOR(AD<double>) ADvector;
    void operator()(ADvector& fg, const ADvector& vars) {
        //  implement MPC
        // `fg` a vector of the cost constraints, `vars` is a vector of variable values (state & actuators)
        // NOTE: You'll probably go back and forth between this function and
        // the Solver function below.

        //if car is going round a sharp bend (ie radius of curvature is small, reduce the target speed)
        if(rCurve < 1) {
            ref_v = 30;
        } else if (rCurve < 10){
            ref_v = 40;
        }else if (rCurve < 80){
            ref_v = 50;
        } else if (rCurve < 200){
            ref_v = 70;
        }else {
            ref_v = 90;
        }
        //init cost
        fg[0] = 0.0;

        //errors in desired state vs actual
        for  (int t = 0; t< N; t++) {
            fg[0] += v_cost_weight * CppAD::pow(vars[v_start + t] -ref_v, 2);
            fg[0] += cte_cost_weight * CppAD::pow(vars[cte_start + t] - ref_cte, 2);
            fg[0] += epsi_cost_weight * CppAD::pow(vars[epsi_start + t] - ref_epsi, 2);
        }

        //minize delta angle changes and acceleration
        for  (int t = 0; t< N -1; t++) {
            fg[0] += delta_cost_weight * CppAD::pow(vars[delta_start+ t], 2);
            fg[0] += accel_cost_weight * CppAD::pow(vars[a_start + t], 2);
        }
        //minize 'jerkiness'
        for  (int t = 0; t< N -2; t++) {
            fg[0] += delta_change_cost_weight * CppAD::pow(vars[v_start] *(vars[delta_start + t + 1] - vars[delta_start + t])/dt, 2);
            fg[0] += accel_change_cost_weight * CppAD::pow((vars[a_start + t + 1] - vars[a_start + t ])/dt, 2);
        }

        //vehicle model:
        fg[1 + x_start] = vars[x_start];
        fg[1 + y_start] = vars[y_start];
        fg[1 + psi_start] = vars[psi_start];
        fg[1 + v_start] = vars[v_start];
        fg[1 + cte_start] = vars[cte_start];
        fg[1 + epsi_start] = vars[epsi_start];

        // The rest of the constraints
        for (int t = 0; t < N-1; t++) {
            //at time t+1:
            AD<double> x1 = vars[x_start + t + 1];
            AD<double> y1 = vars[y_start + t + 1];
            AD<double> psi1 = vars[psi_start + t + 1];
            AD<double> v1 = vars[v_start + t + 1];
            AD<double> cte1 = vars[cte_start + t + 1];
            AD<double> epsi1 = vars[epsi_start + t + 1];

            //  at time = t
            AD<double> x0 = vars[x_start + t ];
            AD<double> y0 = vars[y_start + t];
            AD<double> psi0 = vars[psi_start + t];
            AD<double> v0 = vars[v_start + t];
            AD<double> cte0 = vars[cte_start + t];
            AD<double> epsi0 = vars[epsi_start + t];

            AD<double> delta0 = vars[delta_start + t];
            AD<double> a0 = vars[a_start + t];

            AD<double> f0 = coeffs[0] + coeffs[1] * x1  + coeffs[2] * x1 * x1 + coeffs[3] * CppAD::pow(x1,3);
//            AD<double> f0 = coeffs[0] + coeffs[1] * x0  + coeffs[2] * x0 * x0 + coeffs[3] * x0 * x0 * x0;
            AD<double> psi_desired = CppAD::atan(coeffs[1] + 2 * coeffs[2] * x0 + 3 * coeffs[3] * x0 * x0);

            fg[1 + x_start + t + 1] = x1 - (x0 + v0 * CppAD::cos(psi0) * dt);
            fg[1 + y_start + t + 1] = y1 - (y0 + v0 * CppAD::sin(psi0) * dt);
            fg[1 + psi_start + t + 1] = psi1  - (psi0 + v0  * delta0/Lf * dt) ;
            fg[1 + v_start + t + 1] = v1 - (v0 + a0 * dt);
            fg[1 + cte_start + t + 1] = cte1 -(f0-y0 + (v0 * CppAD::sin(epsi0) * dt));
            fg[1 + epsi_start + t + 1] = epsi1 - (psi0 - psi_desired + v0  * delta0/Lf * dt) ;
        }

    }
};

//
// MPC class definition implementation.
//
MPC::MPC() {}
MPC::~MPC() {}

vector<double> MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs) {
    bool ok = true;
    size_t i;
    typedef CPPAD_TESTVECTOR(double) Dvector;

//    cout << "coeffs at mpc solve"<< coeffs[0] <<","<<coeffs[1]<< ","<< coeffs[2]<< endl;

    double x = state[0];
    double y = state[1];
    double psi = state[2];
    double v = state[3];
    double cte = state[4];
    double epsi = state[5];
    double rCurve = state[6];

//    cout<< "states"<<endl;
//    cout<< "x:" << x << endl;
//    cout<< "y:" << y << endl;
//    cout<< "psi:" << psi << endl;
//    cout<< "v:" << v << endl;
//    cout<< "cte:" << cte << endl;
//    cout<< "epsi:" << epsi << endl;

    // TODO: Set the number of model variables (includes both states and inputs).
    // For example: If the state is a 4 element vector, the actuators is a 2
    // element vector and there are 10 timesteps. The number of variables is:
    //
    // 4 * 10 + 2 * 9
    size_t n_vars = 6 * N  + 2 * (N-1);
    size_t n_constraints = 6 * N;

    // Initial value of the independent variables.
    // SHOULD BE 0 besides initial state.
    Dvector vars(n_vars);
    for (int i = 0; i < n_vars; i++) {
        vars[i] = 0.0;
    }

    vars[x_start] = x;
    vars[y_start] = y;
    vars[psi_start] = psi;
    vars[v_start] = v;
    vars[cte_start] = cte;
    vars[epsi_start] = epsi;

    Dvector vars_lowerbound(n_vars);
    Dvector vars_upperbound(n_vars);

    // Set all non-actuators upper and lowerlimits
    // -- max negative and positive values.
    for (int i = 0; i < delta_start; i++) {
        vars_lowerbound[i] = -1.0e19;
        vars_upperbound[i] = 1.0e19;
    }

    //constraints on delta in radians(25 degrees angle):
    for (int i = delta_start; i < a_start; i++) {
        vars_lowerbound[i] = -0.436332 * Lf;
        vars_upperbound[i] = 0.436332 * Lf;
    }

    // Acceleration/decceleration upper and lower limits.
    for (int i = a_start; i < n_vars; i++) {
        vars_lowerbound[i] = -1.0;
        vars_upperbound[i] = 1.0;
    }

    // Lower and upper limits for the constraints
    // Should be 0 besides initial state.
    Dvector constraints_lowerbound(n_constraints);
    Dvector constraints_upperbound(n_constraints);
    for (int i = 0; i < n_constraints; i++) {
        constraints_lowerbound[i] = 0;
        constraints_upperbound[i] = 0;
    }
    constraints_lowerbound[x_start] = x;
    constraints_lowerbound[y_start] = y;
    constraints_lowerbound[psi_start] = psi;
    constraints_lowerbound[v_start] = v;
    constraints_lowerbound[cte_start] = cte;
    constraints_lowerbound[epsi_start] = epsi;

    constraints_upperbound[x_start] = x;
    constraints_upperbound[y_start] = y;
    constraints_upperbound[psi_start] = psi;
    constraints_upperbound[v_start] = v;
    constraints_upperbound[cte_start] = cte;
    constraints_upperbound[epsi_start] = epsi;


    // object that computes objective and constraints
    FG_eval fg_eval(coeffs, rCurve);

    //
    // NOTE: You don't have to worry about these options
    //
    // options for IPOPT solver
    std::string options;
    // Uncomment this if you'd like more print information
    options += "Integer print_level  0\n";
    // NOTE: Setting sparse to true allows the solver to take advantage
    // of sparse routines, this makes the computation MUCH FASTER. If you
    // can uncomment 1 of these and see if it makes a difference or not but
    // if you uncomment both the computation time should go up in orders of
    // magnitude.
    options += "Sparse  true        forward\n";
    options += "Sparse  true        reverse\n";

    // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
    // Change this as you see fit.
    options += "Numeric max_cpu_time          3.0\n";

    // place to return solution
    CppAD::ipopt::solve_result<Dvector> solution;

    // solve the problem
    CppAD::ipopt::solve<Dvector, FG_eval>(
            options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
            constraints_upperbound, fg_eval, solution);

    // Check some of the solution values
    ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

    // Cost
    //auto cost = solution.obj_value;
    //std::cout << "Cost " << cost << std::endl;

    vector<double> outputs = {};
    outputs.push_back(solution.x[delta_start]);
    outputs.push_back(solution.x[a_start]);

//    cout << "solution solution.x[delta_start]" << solution.x[delta_start]<< endl;
//    cout << "solution solution.x[a_start]" << solution.x[a_start]<< endl;
    for (int t = 0; t< N-1; t++) {
        outputs.push_back(solution.x[x_start+ t + 1]);
        outputs.push_back(solution.x[y_start+ t + 1]);

//        cout << "solution solution.x[x_start+ t + 1]" << solution.x[x_start+ t + 1]<< endl;
//        cout << "solution solution.x[y_start+ t + 1]" << solution.x[y_start+ t + 1]<< endl;
    }
    return outputs;
}
