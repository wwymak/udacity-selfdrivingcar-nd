### Advanced Lane Lines Finding Project

The main task of this project is to detect the left and right lane lines from a video taken from a car. The algorithms/methods
developed from this task can then be adapted for self driving cars to ensure it stays inside a lane.

---

#### Detection steps:
1. calibrate the camera
2. Clean images with Gaussian blur`cv2.GaussianBlur` and also enhance contrast with histogram equalisation `cv2.createCLAHE`
3. correct images from the camera according to the calibration
4. Apply thresholding with color and gradient transforms to isolate lane lines
5. Transform to birds eye view, detect lines and fit to polynomial, calculate radius of curvature
6. Transform lane lines to original viewpoint and draw lines on image
7. Output image as part of video processing pipeline

---

#### Camera Calibration

The camera caibration is done with a set of chessboard images. These images have 9 x 6 chessboard grids-- the grid points
are used as object points, whereas the actual points of these corners (ie image points) in the images are found with the `cv2.findChessboardCorners()`  function. If the image points are found successfully in a calibration image, these points are
appended  to the img_points_arr array and a copy of the object points appended to obj_points_arr array. The
calibration matrices is calculated with `cv2.calibrateCamera()` using these two array as inputs.

Image correction is carried out with the `cv2.undistort()` function.

E.g.

Uncorrected                |  Corrected                |
:-------------------------:|:-------------------------:|
![](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Advanced-Lane-Lines/test_images/test1.jpg)  | ![](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Advanced-Lane-Lines/output_images/test1_camera_corrected.jpg)|
![](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Advanced-Lane-Lines/camera_cal/calibration1.jpg)|![](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Advanced-Lane-Lines/output_images/calibration1.jpg)

Other examples of correct chessboard images are in `https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Advanced-Lane-Lines/output_images/calibration*.jpg` and
examples of the test images after undistortion are in `https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Advanced-Lane-Lines/output_images/test*.jpg`

---

#### Image denoising/enhancing
An example of this step is shown below:

![](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Advanced-Lane-Lines/output_images/image_cleaning.png)

---

#### Thresholding investigations

**Color Thresholding**

However, it is not sufficient to only use color thresholding, as shown below-- in certain conditions, e.g. lots of shadows,
the color thresholding on it's own doesn't isolate the lane lines very well

![](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Advanced-Lane-Lines/output_images/color_thresholding_example_all.jpg)


_Combining HLS and RGB thresholding:_

After experimenting with thresholding either the S channel of the image in HLS color space and the R channel in RGB color space,
I also tried combining them, which seems to give the best result.

**Gradient Thresholding**

To highlight lane lines, I also apply gradient thresholding with the Sobel operators (using `cv2.Sobel`, which takes the gradient of
    an image in either the x or the y direction). The three thresholds I used in combination was
- magnitude thresholding (ie )
- direction thresholding
- value thresholding in both x and y directions

The result of the thresholding can be seen below:

![](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Advanced-Lane-Lines/output_images/sobel_thresholding_in_action.png)

**Thresholding-- combination**

To take advantage of both the gradient and color thresholding, I combined the result of applying the color thresholding Pipeline
and the gradient thresholding pipeline such that the image pixel is 1 if either the color or the gradient threshold returns 1, and zero otherwise:
```
def combined_thresholding(sobel_bin, color_binary):   
    combined = np.zeros_like(sobel_bin)
    combined[(sobel_bin == 1) | (color_binary == 1)] = 1
    return combined
```

The parameters I found to be the best for the thresholding pipeline are:

| Param        | Threshold   |
|:-------------:|:----------:|
| sobel kernel size | (3,3)    |
| sobel_x       | (15,90)      |
| sobel_y       | (20, 90)     |
| sobel magitude | 960, 720    |
| sobel direction| 960, 720    |
| HLS threshold     |(70,255)        |
| RGB threshold     |(150,255)        |


#### Warp perspective
After thresholding the image to enhance the lane lines, the perspective is warped so it seems the lines are viewed from above.
This enables a polynomial to be fitted to the lines. The perspective warping calibration is done by calling `cv2.getPerspectiveTransform`.
on an image with straight lane lines, finding the source points on that image, then the destination points in a topview image, and
calculating the transformation matrices between them. The following images illustrate this process:

![](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Advanced-Lane-Lines/output_images/perspective_transform1.png)
![](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Advanced-Lane-Lines/output_images/perspective_transform2.png)

An thresholded image that is warped to top view (with `cv2.warpPerspective`) looks like:



#### Fit polyline to lanes


#### Reproject lines back to original image
The warp perspective calibration yields 2 matrices: M for converting the normal image to topview, and Minv of converting
the topview image to the normal view. After the lane lines are found, they are drawn onto the image with `cv2.polylines`
and also the polygon these lines form are filled in with `cv2.fillPoly`. The image is then reprojected

#### Radius of curvature/ car offset calculations:
The final step in the processing pipeline is to calculate the radius of curvature of the left and right lines, and also, how far is the car is from the center of the lines. This information can be used in e.g. steering the car back to the center of the lane, indicating
how much turning needs to be applied.

The radius of curvature of the lines (represented by equations `f(y)=Ay^2 + By + C`) are calculated by:

`Rcurv = (1+(2Ay+B)^2)^1.5)/abs(2 * A)`

where the distances are converted to from pixel to actual values first.  In the video pipeline discussed in the following, this information
is printed on the left of each frame of the video



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

The final output video is [here](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Advanced-Lane-Lines/project_video_out_pipeplinev3.mp4)

---


### Usage/running the code:
- The relevant analysis/ investigations are all in the [Lane Lines Project.ipynb](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Advanced-Lane-Lines/Lane%20Lines%20Project.ipynb) file
-  To only run the lane line pipeline, you can use laneline_pipeline.py file like so: `python laneline_pipeline.py --input input_video project_video.mp4  --output project_video_out.mp4`

---

### Further investigations
- improve pipeline to work on the challenge videos-- these are the ones with a lot more light and shadow (as well as a less evenly
    colored road surface) as compared to the project video
at the moment, it sort of works on the challenge_video but the
pipeline does have a tendency to detect the edges of the road as the left lane line. Potentially tuning the thresholding params
could help.
