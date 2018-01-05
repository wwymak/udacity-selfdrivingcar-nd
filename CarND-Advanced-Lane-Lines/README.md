### Advanced Lane Lines Finding Project

The main task of this project is to detect the left and right lane lines from a video taken from a car. The algorithms/methods
developed from this task can then be adapted for self driving cars to ensure it stays inside a lane.

---

#### Detection steps:
1. calibrate the camera
2. correct images from the camera according to the calibration
3. Apply thresholding with color and gradient transforms to isolate lane lines
4. Transform to birds eye view, detect lines and fit to polynomial, calculate radius of curvature
5. Transform lane lines to original viewpoint and draw lines on image
6. Output image as part of video processing pipeline

---

### Camera Calibration

The camera caibration is done with a set of chessboard images. These images have 9 x 6 chessboard grids-- the grid points
are used as object points, whereas the actual points of these corners (ie image points) in the images are found with the `cv2.findChessboardCorners()`  function. If the image points are found successfully in a calibration image, these points are
appended  to the img_points_arr array and a copy of the object points appended to obj_points_arr array. The
calibration matrices is calculated with `cv2.calibrateCamera()` using these two array as inputs.

Image correction is carried out with the `cv2.undistort()` function.

E.g.

Uncorrected                |  Corrected                |
|:-------------------------:|:-------------------------:|
|![](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Advanced-Lane-Lines/test_images/test1.jpg)  |  |![](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Advanced-Lane-Lines/output_images/test1_camera_corrected.jpg)|![](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Advanced-Lane-Lines/camera_cal/calibration1.jpg)  |
|![](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Advanced-Lane-Lines/output_images/calibration1.jpg)|

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 585, 460      | 320, 0        |
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

#### Pipeline (video)

The processing steps for the video is quite similar, however, there are a few extra steps to smooth out the line detection:
* the last 5 frames are averaged to find the line of best fit for the lane lines in the current frame (this helps to reduce 'jitter'
in the detection). The values for the previous fits are stored in a 'Line' class instance, one for the left lane and one for the right
* if the separation of the left and right lines are much smaller or larger than 3.7m, reject the detection and
use the previous value instead (unless it's the first image)
* if the radius of curvature of the left line is less than half that of the right (or the right curvature less than half
    that of the left), reject the detection

The final output video is [here](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Advanced-Lane-Lines/project_video_out_pipeplinev3all.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

### Usage/running the code:
- The relevant analysis are all in the [Lane Lines Project.ipynb](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Advanced-Lane-Lines/Lane%20Lines%20Project.ipynb) file
-  
