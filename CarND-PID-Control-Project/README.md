## PID controller project
--

This project is an implementation of a [PID controller](https://en.wikipedia.org/wiki/PID_controller). In the project, 
PID controllers are used to tune the steering angle and the throttle of the car in the simulator so it goes at a target speed
and minimizes it's deviations from it's path.

In this scenario of the car, the desired steady state when driving is a zero cross track error (cte -- ie, distance from 
center of the road/lane) and if using a throttle control, the target velocity. 

A PID controller
is a simple but effective way of correcting deviations from a set point. It requires no detailed knowledge of the model behind the
device. Instead, it provides an opposing signal to bring the system back to the setpoint. There are 3 terms that make up the
output of a PID controller-- the proportional term  (Kp), derivative term (Kd), and the integral term (Ki).

The following equation (source: [wikipedia](https://en.wikipedia.org/wiki/PID_controller)) expresses what the 3 terms does:

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/69072d4013ea8f14ab59a8283ef216fb958870b2)



//todo** explain what are the contributions fo the 3 terms



---
### Tuning PID parameters
The parameters were manually tuned until the car has the desired behaviour of staying on the track. A good overview 
 of the methods can be found [here](https://en.wikipedia.org/wiki/PID_controller#Overview_of_methods). For manual
 tuning, the article suggest that the Kp term should be increased until the system oscillates, and the Kd term increased.
 However, the car in the simulation starts oscillating with even a minimal application of Kp, and wanders off the track 
 if Kd is not applied, so I adapted the manual tuning method-- 
   1. I increased the Kp term in small increments so that the car is fairly stable in the first straight part of the track. (the car then starts oscillating
   around center of the road as it travels along) 
   2. Increased the Kd term (which helps in preventing this overshoot) until the car stays on the track the whole way round. 
   3. In the lectures it suggests using the Ki term as well to prevent systematic drift due to any leftover errors.
   In this case, I found this term unecessary to keep the car driving successfully round the track.
   
   A good set of parameters I found for the PID controller for the driving angle is 
   > Kp = 0.15 Ki = 0 Kd = 1.6
   
To make the car drive faster, I also used a PID controller for the throttle to keep the speed at the target 60mph, with 
the following condition-- if the error from road center is starting to get high, slow the car down (throttle = 0), and if it's really high, 
brake (throttle < 0) :  
```
if (fabs(cte) > 1.5) {
    throttle_val = 0;
}
// car oscillating far too much, have to brake
if (fabs(cte) > 5.0) {
    throttle_val = -1.0;
}
```
*note* the threshold cte parameter values at which to reduce the throttle are tuned manually as well

A good set of parameters I found for the PID controller for the throttle is (using above params for steering angle)
   > Kp = 0.1 Ki = 0 Kd = 0.5

----

### Project output


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

#### Thoughts and future work
Make the car go faster:
- more fine tuning of pid parameters might 
Autotuning:
- there are various methods/papers around autotuning PID controllers. However, most of them seems to require
an artificial stimulation of the car controls and measure it's behaviour, which I find unintutive to try in this case 
since if the car drives off the track in the simulation it's hard to have any meaningful measurements. 
