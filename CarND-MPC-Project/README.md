## Model Predictive Control Project
The aim of this project is to use a model predictive control 
algorithm to drive a car round a track in a simulator.
The algorithm provides steering angle and throttle
commands to the car by calculating what the
cross track errors are currently, and based on the vehicle movement
model, how it should move to minimize this error.

---
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