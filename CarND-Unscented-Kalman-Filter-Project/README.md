## Unscented Kalman Filter Project

This project is a C++ implementation of an [Unscented Kalman
Filter](https://en.wikipedia.org/wiki/Kalman_filter#Unscented_Kalman_filter),
applied to noisy lidar and radar measurements of a bicycle in the simulatro 

Similar to the Extended Kalman Filter (EKF), the Unscented Kalman Filter (UKF)
is for handling non linear measurements space, with the difference being how it models the
non linearity. 

The EKF uses the constant veloctiy model whereas the UKF uses the constant
velocity and turn rate model, which is generally more accurate as objects don't generally go in straight lines all the time.

In the UKF, rather than using a linear approximation of a non linear function, it uses a sampling of points
from a Gaussian distribution an use these in the non linear equations for the radar measurements. The steps for the UKF algorithm 
is shown in the following diagram (credit: udacity) 

![The steps in a UKF algorithm (credit: udacity)](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Unscented-Kalman-Filter-Project/UKF.png)

---
### Project code
These are in the src directory.

The `ukf.cpp` is the class that implements the actual filter predict and update steps. The lidar measurements 
are linear with respect to the coordinate space, so a normal Kalman Filter update is used, whereas for the radar measurements,
 the non-linear relation between the measurements and the coordinate space means that a non linear approach, 
 in this case the Unscented Kalman Filter is needed.

`tools.cpp` file contains 2 helper functions -- one for calculating the root mean squared error (RMSE) 
values between the predicted and the ground truth values,and one for normalising the angles to between pi and -pi for 
C++ functions.

`main.cpp` communicates with the simulator, passes the values from the simulation to the UKF code, 
and sends the estimated locations from the filter back to the simulator. There is also an extra bit where the output 
is written to a csv file for visualisations.

#### Parameter Tuning
Starting with sensible guess for process noise can help the filter converge to the ground truth value a lot faster. In this
case, the relevant are the standard deviation of longitudinal acceleration and the standard deviation of the yaw rate. 
As the vehicle in this case is a bicycle, I set these to be 0.2 m/s and 0.2 rad/s.

e.g. in a bicycle travelling at 20kph (~6m/s), it can get to that speed in 30s, which is not unreasonable.

#### Visualisations:
The output of the filter at each timestep is also written to a file so the outputs can be visualised/analysed separately. 
The visualisations are in the `output-vis.ipynb` notebook. (if not running notebook locally, dyanmic charts viewable 
here](http://nbviewer.jupyter.org/github/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Unscented-Kalman-Filter-Project/output_vis.ipynb)))

--- 
### Running the code:
This assumes the term 2 simulator has been installed and the relevant
packages setup

Then in the project root directory, run:

1. mkdir build
2. cd build
3. cmake ..
4. make
5. ./UnscentedKF
6. Launch simulator and see how the green(predictions) matches 
against the measurments

---

### UKF outputs
The RMSE outputs of this implementation on the 'obj_pose-laser-radar-synthetic-input.txt' dataset is

```
RMSE values:   
x: 0.0796  
y: 0.0980  
vx: 0.2966  
vy: 0.2585
```

As expected, using only one sensor gives a less accurate localisation of the vehicle:


Radar only:

```
RMSE values:   
x: 0.1982  
y: 0.2280  
vx: 0.3608  
vy: 0.03485
```

Laser only:

```
RMSE values:   
x: 0.1293  
y: 0.1146  
vx: 0.4857  
vy: 0.2609
```

Compared to the output from the previous project on Extended Kalman Filters,
the output from the unscented Kalman Filter is also more accurate for all 
3 different measurements (both sensors, laser only, lidar only.) The RMSE 
values from the Extended Kalman Filter ran on the same data can be seen [here](https://github.com/wwymak/udacity-selfdrivingcar-nd/tree/master/CarND-Extended-Kalman-Filter-Project/readme.md)
