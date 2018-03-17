## Unscented Kalman Filter Project

This project is a C++ implementation of an Unscented Kalman
Filter 


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