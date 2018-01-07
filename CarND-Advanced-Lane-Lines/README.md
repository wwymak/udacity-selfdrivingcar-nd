## Advanced Lane Lines Finding Project

The main task of this project is to detect the left and right lane lines from a video taken from a car. The algorithms/methods
developed from this task can then be adapted for self driving cars to ensure it stays inside a lane.

---

### Detection steps:
1. calibrate the camera
2. Clean images with Gaussian blur`cv2.GaussianBlur` and also enhance contrast with histogram equalisation `cv2.createCLAHE`
3. correct images from the camera according to the calibration
4. Apply thresholding with color and gradient transforms to isolate lane lines
5. Transform to birds eye view, detect lines and fit to polynomial, calculate radius of curvature
6. Transform lane lines to original viewpoint and draw lines on image
7. Output image as part of video processing pipeline

---

### Camera Calibration

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

### Image denoising/enhancing
An example of this step is shown below:

![](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Advanced-Lane-Lines/output_images/image_cleaning.png)

---

### Thresholding investigations

**Color Thresholding**

Lane lines are white in color (and also could be yellow as per the input video in the US). This is distinct from the road surface
and the surroundings, so it should be possible to isolate the line pixels to a certain degree with color thresholding. The most promising
color spaces are RGB (using the R color channel), and HLS space (using the S space-- the lines reflect light differently to the
    surroundings so should be able to isolate them with the Saturation channel)

_Combining HLS and RGB thresholding:_

After experimenting with thresholding either the S channel of the image in HLS color space and the R channel in RGB color space,
I also tried combining them, which gives the best result. (as shown in the series of images below, which is from the color thresholding pipeline applied to the test images, the 'combined' color thresholding method gives images with the least noise )

However, it is not sufficient to only use color thresholding-- as can be seen in the images, in certain conditions, e.g. lots of shadows,
the color thresholding on it's own doesn't isolate the lane lines very well, e.g. missing some lane pixels, or confusing light and shadow with lane lines

![](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Advanced-Lane-Lines/output_images/color_thresholding_example_all.jpg)



**Gradient Thresholding**

To highlight lane lines, I also apply gradient thresholding with the Sobel operators (using `cv2.Sobel`, which takes the gradient of
    an image in either the x or the y direction). The three thresholds I used in combination was
- magnitude thresholding (ie threshold on the value of ((sobelx)^2 + (sobely)^2) ^ 0.5)
- direction thresholding (so only allow features that have a certain angle in the image corresponding to typical angles a lane line would
    have in a camera. Gradient angle is obtained by arctan(sobely/sobelx))
- value thresholding in both x and y directions (so threshold sobelx and sobely separately, based on the absolute values of each)


The result of the sobel thresholding can be seen below:

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
This method does not really fix the color thresholding issue where it grabs too many positive pixels due to high light intensity variations, but it does fix the cases where a line might be missing from the image after color thresholding, so I can apply
a more aggressive color threshold to remove some of the noise in the color space and supplement missing detections with those from
the gradient thresholding space.

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

The result of the combination thresholding can be seen below:

![](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Advanced-Lane-Lines/output_images/combined_thresholding_example_all.jpg)


### Warp perspective
After thresholding the image to enhance the lane lines, the perspective is warped so it seems the lines are viewed from above.
This enables a polynomial to be fitted to the lines. The perspective warping calibration is done by calling `cv2.getPerspectiveTransform`.
on an image with straight lane lines, finding the source points on that image, then the destination points in a topview image, and
calculating the transformation matrices between them. The following images illustrate this process:

![](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Advanced-Lane-Lines/output_images/perspective_transform1.png)
![](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Advanced-Lane-Lines/output_images/perspective_transform2.png)

The source and destination points used are:

| Source        | Destination   |
|:-------------:|:-------------:|
| 205, 725      | 205, 725      |
| 450,550       | 205,550       |
| 842,550       | 1115,550     |
| 1115, 725     | 1115, 725    |

I chose not to change the y values of the warped to topview images since this would affect the radius of curvature calculation--
e.g. if my destination points are chosen to be [(x1, 0), (x1, ymax), (x2, 0), (x2, ymax)] then the lines would be 'strecthed' in the vertical direction and the radius of curvature would be less than it should be (of course, I could have added in extra adjustment but
    decided to keep things simple)

An thresholded image that is warped to top view (with `cv2.warpPerspective`) looks like:

![](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Advanced-Lane-Lines/output_images/warp_perspective.jpg)

### Finding lane line pixels in the reprojected image-- histogram method
1. find where the peaks are across a line in the middle of the image by plotting a histogram:

    ![](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Advanced-Lane-Lines/output_images/lane_finding1.png)

2. Find the left and right peaks using the argmax function in numpy, then searching in a series of windows from top to bottom of the image,
find where the non zero pixels in the binary image is. Each subsequent window takes into account the position of the previous one-- the peak position is updated in each window and the next one will use this to search for non zero pixels:

![](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Advanced-Lane-Lines/output_images/lane_finding2.png)

3. the nonzero pixels form the points for the polynomial fit (order2):

![](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Advanced-Lane-Lines/output_images/lane_finding3.png)

### Fit polyline to lanes
Once the lane line pixels are found, a 2nd order polynomial is fitted to the points with the `np.polylines` function


### Reproject lines back to original image
The warp perspective calibration yields 2 matrices: M for converting the normal image to topview, and Minv of converting
the topview image to the normal view. After the lane lines are found, they are drawn onto the image with `cv2.polylines`
and also the polygon these lines form are filled in with `cv2.fillPoly`. The image is then reprojected back onto the original view
with Minv

### Radius of curvature/ car offset calculations:
The final step in the processing pipeline is to calculate the radius of curvature of the left and right lines, and also, how far is the car is from the center of the lines. This information can be used in e.g. steering the car back to the center of the lane, indicating
how much turning needs to be applied.

The radius of curvature of the lines (represented by equations `f(y)=Ay^2 + By + C`) are calculated by:

`Rcurv = (1+(2Ay+B)^2)^1.5)/abs(2 * A)`

where the distances are converted to from pixel to actual values first.  In the video pipeline discussed in the following, this information
is printed on the left of each frame of the video. In the topview image, the values for the left and right lane lines should be roughly similar, and on the order of a few kms. In the video pipeline, I have set a check such that if either the left or right curvature is less  than half of the other, the detection pipeline hasn't gone well and the values are not used in that frame. On some of the test images, the radius of curvature of the left and right lanes differ by much more than half, but the detected lane areas actually are still fairly accurate.
![](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Advanced-Lane-Lines/output_images/lane_finding_pipeline_alltests.jpg)

The following image shows the final output from the whole pipeline after reprojection and radius of curvature calculation on one of the frames of the project video:

![](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Advanced-Lane-Lines/output_images/example_output.jpg)


---

### Pipeline (video)

The processing steps for the video is quite similar, however, there are a few extra steps to smooth out the line detection:
* the last 5 frames are averaged to find the line of best fit for the lane lines in the current frame (this helps to reduce 'jitter'
in the detection). The values for the previous fits are stored in a 'Line' class instance, one for the left lane and one for the right
* if the separation of the left and right lines are much smaller or larger than 3.7m, reject the detection and
use the previous value instead (unless it's the first image)
* if the radius of curvature of the left line is less than half that of the right (or the right curvature less than half
    that of the left), reject the detection

The final output video is [here](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Advanced-Lane-Lines/project_video_output_v4.mp4)

---


### Usage/running the code:
- The relevant analysis/ investigations are all in the [Lane Lines Project.ipynb](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Advanced-Lane-Lines/Lane%20Lines%20Project.ipynb) file
-  To actually run the final lane line detection pipeline on videos without having to go through the notebook, you can use laneline_pipeline.py file like so: `python laneline_pipeline.py --input  project_video.mp4  --output project_video_out.mp4`. The code
in the file is also annotated to point out where the processing steps are
- helpers.py contains a few useful functions (could probably have put the relevant one into laneline_pipeline.py, but it works where it is now :wink:)

---

### Further investigations
- improve pipeline to work on the challenge videos-- these are the ones with a lot more light and shadow (as well as a less evenly
    colored road surface) as compared to the project video
at the moment, it half  works on the challenge video (see [challenge_video_out_pipeline_v3all.mp4](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Advanced-Lane-Lines/challenge_video_out_pipeline_v3all.mp4)) but the
pipeline does have a tendency to detect the edges of the road as the left lane line. Potentially tuning the thresholding params
could help.
- optimisation: at the moment, it takes around 15 minutes to process a one minute video. This is hardly ideal as the processing time means
that the line detection can't happen in real time, which is needed for a self driving car.
