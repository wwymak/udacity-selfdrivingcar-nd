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

The Kp term is proportional to the cross check error, so when the error increases, the Kp term helps to drive
the controller to make the car go the opposite way.

However, the Kp term is linear, so it causes the system to overshoot it's set point and oscillate. This is 
reduced by the Kd term, which is proportional to the change in error-- when the change in error drops, the corrective
forces also drops, which reduces the amount by which the controller overshoots.

The Ki term may or may not be necessary-- it helps in removing any errors that remain after the Kp and Kd terms are applied.
But since it gets bigger and bigger over time if small amounts of error remain, it can actually make the oscillations worse as 
the overall corrective term gets large.

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
   > Kp = 0.15 Ki = 0 Kd = 1.8
   
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
Can be viewed at this [youtube link](https://youtu.be/RHsDqGic7qc)

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
5. ./pid (with the default good pid params) _or_ e.g. `./pid 0.15 0.0 2.0 0.25 0 0.7 ` if you want to 
supply your own pid params-- the first 3 terms are the Kp, Ki, Kd terms for the steering angle control, 
the last 3 terms the Kp, Ki, Kd terms for setting the throttle to get to the desired speed
6. Launch simulator and see how the green(predictions) matches 
against the measurments

---

#### Thoughts and future work

#####Make the car go faster:
- more fine tuning of pid parameters might help the car stay on track better at a higher speed-- at the current parameters,
setting a target speed higher than the 60mph causes the car to oscillate at a higher amplitude and ended up in it veering 
off the road

#####Autotuning:
- there are various methods/papers around autotuning PID controllers. However, most of them seems to require
an artificial stimulation of the car controls and measure it's behaviour, which I find unintutive to try in this case 
since if the car drives off the track in the simulation it's hard to have any meaningful measurements. I also tested 
an implementation of the Twiddle algorithm, with each iteration corresponding to a certain number of timesteps.

However, the algorithm assumes you can go round the same set of 
conditions, ie starting from the start point, which I would not be able to do automatically.

It would also be interesting to explore newer tuning methods such as [this](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5038707/)
neural network based PID controller.

##### Online manual tuning
It would be interesting to make an application that can sit between the PID controller and the simulator that 
can listen on the signals between them and use this to do real time charting of the cte errors. Also, if I can use 
this to feed in an extra manual controls to change the PID params this would be really useful for tuning.
