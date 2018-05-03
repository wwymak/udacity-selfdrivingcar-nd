## Model Predictive Control Project
The aim of this project is to use a model predictive control 
algorithm to drive a car round a track in a simulator.
The algorithm provides steering angle and throttle
commands to the car by calculating and minimizing the
difference between the actual and desired state are currently, and based on the vehicle movement
model, how it should move to minimize this error.

The result of running my implementation of the model can be viewed [at this link](https://youtu.be/yT2Td2zRNEE)

The model predictive algorithm is more complex than a PID controller as optimisation is carried out at each timestep.
At each timestep, the model tries to predict what the next N states should be, then it moves to the 1st state, check for 
errors, updates the steps in should take based on these errors, and repeats. These predictions are based on a physical model
for the system, e.g. the kinematic model for a car in this case.

#### Vehicle model
The motion of the car is modelled as:

```
px += v * cos(psi) * dt;
py += v * sin(psi) * dt;
psi -= v/Lf * dt *  steer;
v += acceleration * dt;

```

where (px, py) is the position of the car in world space, v the speed of the car, psi the heading direction, Lf a corrective
constant for handling the fact that the car's center of mass and the tyres are not in the same place, steer the steering angle.

#### Code implemetation
The code for handling the car simulation is mostly in main.cpp and MPC.cpp, with a few utility functions in utils.h.

The main steps are:

1. messages are received from the simulator about the upcoming waypoints, current position, speed, throttle and steering angle of the car.
2. convert the waypoints from world space to car space (to simplify the calculations)
3. convert the speed of the car  from miles per hour (from simulator) to meters per second (to match the metric units of the waypoints)
4. fit a polynomial to the waypoints -- I used a polynomial of degree 3 as it fits the waypoints smoothly 
5. calculate the cross track error, heading error and the radius of curvature of the predicted path at the position of the car,
and update the state of the car
6. use the optimisation solver to solve for the best steering angle and throttle to control the car
7. send these messages back to the simulator

#### Optimisation costs
Different weights are applied to the various errors as below. In my code, the values for all
the weights are fairly similar, with most of them being 1 with the exception of the cross track error and the 
heading error (5.0)-- I found that tuning these params did not help a lot with the car driving correctly, and that
the most important factor in finally getting the car to drive fairly fast but stay in the track is actually the handler 
for turns (see next section)
 
```
// minimizing the actual errors:
for (int t = 0; t< N; t++) {
    //error in velocity
    fg[0] += v_cost_weight * CppAD::pow(vars[v_start + t] -ref_v, 2);
    //error in cross track error
    fg[0] += cte_cost_weight * CppAD::pow(vars[cte_start + t] - ref_cte, 2);
    /error in heading
    fg[0] += epsi_cost_weight * CppAD::pow(vars[epsi_start + t] - ref_epsi, 2);
}

//minize delta angle changes and acceleration
for  (int t = 0; t< N -1; t++) {
    fg[0] += delta_cost_weight * CppAD::pow(vars[delta_start+ t], 2);
    fg[0] += accel_cost_weight * CppAD::pow(vars[a_start + t], 2);
}
//minize 'jerkiness': so the accceleration and angle change doesn't change too much between each timestep
for  (int t = 0; t< N -2; t++) {
    fg[0] += delta_change_cost_weight * CppAD::pow(vars[v_start] *(vars[delta_start + t + 1] - vars[delta_start + t])/dt, 2);
    fg[0] += accel_change_cost_weight * CppAD::pow((vars[a_start + t + 1] - vars[a_start + t ])/dt, 2);
}
```
#### Handling turns
To go round turns safely, the car has to slow down, and the model has to
handle the turns slightly different. To calculate how 'curved' the 
current track is, I calculated the radius of curvature (R) of the predicted
path via 

> R = (1 + (dy/dx)^2) ^ 1.5/ (d2y/dx2)

where dy/dx is the first order derivative of the path polynomial equation, and
d2y/dx2 the second order derivative.

I modified the target velocity in the MPC controller class such that 
```
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
```

where `rCurve` is the radius of curvature, and ref_v the target velocity. The setpoints of rCurve in which to
reduce the velocity is found by manual tuning.

#### Handling Latency
In a real car, there is going to be small latency between the signal from the controllers to the 
mechanical parts moving the car. This means the car will have moved
by the time the signal the steering mechanism. To compensate for this, I calculated what the 
vehicle state _should_ be after the latency before passing it into the MPC predition (this is done via the 
`estimateState` function in utils.cpp) 

##### choosing values for N(the number of steps to look ahead), dt(time in seconds between each step in the prediction)
In each timestamp of prediction, te MPC controller tries to predict what is going to happen in the next N steps. Choosing
the right value for N and dt is important to getting the car running as too many steps is computationally expensive and
the controller might be too slow in prediction (and this may also cause convergence issues in the error optimsation), while
too little steps might mean that the controller don't have a good estimation of the next states. In my code, I chose N = 25 and
dt = 0.05, as this combination gives the best results after a trial and error process. 


----

#### Running the code:
Ipopt is required to solve the
vehicle model problem for this project, however, different installations of it
 seems to have a huge impact on whether it works or not. To make
 sure the code work reliably it is best to use a docker image with
 the relevant dependences. 
1. make sure you have docker installed and running
2. get a copy of the term 2 docker image from docker hub: 
>`docker pull udacity/controls_kit`
3. `docker run -it -p 4567:4567  -v /path/to/my/project/directory:/mpc -w /mpc udacity/controls_kit /bin/bash`
4. in the docker container, make sure you have mounted the code directory,
and in it, do
5. `mkdir build && cd build`
6. `cmake..`
7. `make`
8. open the simulator and run `./mpc`