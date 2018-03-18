## Unscented Kalman Filter Project

This project is a C++ implementation of an [Unscented Kalman
Filter](https://en.wikipedia.org/wiki/Kalman_filter#Unscented_Kalman_filter),
applied to noisy lidar and radar measurements of a bicycle in the simulatro 

Similar to the Extended Kalman Filter, the Unscented Kalman Filter
is for handling non linear measurements space, with the difference being how it models the
non linearity. 

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
