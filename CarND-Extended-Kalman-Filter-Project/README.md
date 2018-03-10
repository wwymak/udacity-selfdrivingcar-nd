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

----

### Project code

These are in the `src` directory. the `kalman_filter.cpp` is the class
that implements the actual 

Visualisations: the output of the filter at each timestep is also 
written to a file so the outputs can be visualised/analysed separately.
The visualisations are in the `EKF-output-vis.ipynb` notebook. 
(if not running notebook locally, dyanmic charts viewable [here](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Extended-Kalman-Filter-Project/EKF-output-vis.ipynb))

#### Running the code:
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

### EKF outputs
The RMSE outputs of this implementation on the 'obj_pose-laser-radar-synthetic-input.txt' dataset is
```
x: 0.1029  
y: 0.1024  
vx: 0.4582  
vy: 0.5152  
```
which satisfied the project requirements of `RMSE <= [.11, .11, 0.52, 0.52]` 
