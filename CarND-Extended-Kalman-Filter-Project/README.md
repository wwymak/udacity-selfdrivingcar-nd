## Extended Kalman Filter Project

This project is a C++ implementation of an [Extended Kalman Filter](https://en.wikipedia.org/wiki/Extended_Kalman_filter),
applied to noisy lidar and radar measurements of a position of 
a bicycle. 

Kalman Filters applied in the context of a self driving car, are algorithms
that can take in noisy sensor data and convert them into accurate
predictions of the position of objects those sensors are trying to track.
The filter goes by a series of update/ predict steps that make the measurement 
more accurate over time. 

The Extended Kalman Filter is a version of the Kalman filter that
can handle non-linear measurements. In this case, the non-linear 
measurment comes from the radar measurement, whereas the lidar measurements are
linear.

The steps in a kalman filter algorithm that is implemented in the code is shown in the following diagram (credit: udacity)
![The steps in a kalman filter algorithm (credit: udacity)](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Extended-Kalman-Filter-Project/kalman_filter_algo.png)

----

### Project code

These are in the `src` directory. 

The `kalman_filter.cpp` is the class
that implements the actual filter predict and update steps. The lidar measurements
are linear with respect to the coordinate space, so a normal Kalman Filter update is used, whereas
for the radar measurements, the non-linear relation between the measurements and the coordinate
space means that a non linear approach, in this case the Extended Kalman Filter is needed.

The `FusionEKF.cpp` file implements the sensor fusion process-- at each timestep, it  predict, then
update based on the values it receives from the sensors, calling the relevant function in the kalman filter class depending on the
which sensor the measurment is from

`tools.cpp` file contains 2 helper functions -- one for calculating the root mean squared error (RMSE) values between the
predicted and the ground truth values,and one for calculating the Jacobian matrix which is needed in the non-linear EKF 
algorithm.

`main.cpp` communicates with the simulator, passes the values from the simulation to the fusionEKF code, and sends the estimated
locations from the filter back to the simulator. There is also an extra bit where the output is written to a csv file for 
visualisations.

---

##### Visualisations: 

The output of the filter at each timestep is also 
written to a file so the outputs can be visualised/analysed separately.
The visualisations are in the `EKF-output-vis.ipynb` notebook. 
(if not running notebook locally, dyanmic charts viewable [here](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Extended-Kalman-Filter-Project/EKF-output-vis.ipynb))

--- 
### Running the code:
This assumes the term 2 simulator has been installed and the relevant
packages setup

Then in the project root directory, run:

1. mkdir build
2. cd build
3. cmake ..
4. make
5. ./ExtendedKF
6. Launch simulator and see how the green(predictions) matches 
against the measurments

---

### EKF outputs
The RMSE outputs of this implementation on the 'obj_pose-laser-radar-synthetic-input.txt' dataset is
```
x: 0.1029  
y: 0.1024  
vx: 0.4582  
vy: 0.5152  
```
which satisfied the project requirements of `RMSE <= [.11, .11, 0.52, 0.52]` 

By commenting out the relevant parts of the code, we can also get
the RMSE values of using just one of the sensors. From the RMSE, we can see that the laser is much more accurate
than the radar, which is expected. But we can see that using both together gives a more accurate estimation of the position,
hence why we're using sensor fusion :nerd_face:
 
Laser only:
```
x: 0.1473  
y: 0.1153  
vx: 0.6383  
vy: 0.5346 

```

Radar only:
```
x: 3.0325  
y: 5.5079  
vx: 1.6631  
vy: 2.4571

```