## PID controller project
--

This project is an implementation of a [PID controller](https://en.wikipedia.org/wiki/PID_controller). In the project, 
PID controllers are used to tune the steering angle and the throttle of the car in the simulator so it goes at a target speed
and minimizes it's deviations from it's path.

A PID controller
is a simple but effective way of correcting deviations from a set point. It requires no detailed knowledge of the model behind the
device. Instead, it provides an opposing signal to bring the system back to the setpoint. There are 3 terms that make up the
output of a PID controller-- the proportional term, derivative term, and the integral term.



----

### Project code

These are in the `src` directory. 

The `PID.cpp` is the class
that implements an instance of the PID controller 

`main.cpp` communicates with the simulator, passes the values from the simulation to the PID controllers, and sends steering
and throttle instructions back to the car

--- 
### Running the code:
This assumes the term 2 simulator has been installed and the relevant
packages setup

Then in the project root directory, run:

1. mkdir build
2. cd build
3. cmake ..
4. make
5. ./pid 
6. Launch simulator and see how the green(predictions) matches 
against the measurments

---
