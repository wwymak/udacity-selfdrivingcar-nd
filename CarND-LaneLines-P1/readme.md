# **Finding Lane Lines on the Road**

This project is to implement a simple computer vision algorithm for finding the left and right
lane Lines from photos/videos that a self driving car should follow. Other than the standard
numerical tools of numpy, the only other library needed is opencv3 for computer vision tasks.

The code for processing the test images and the videos are in [P1.ipynb](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-LaneLines-P1/P1.ipynb)

The output videos are in the [test videos output](https://github.com/wwymak/udacity-selfdrivingcar-nd/tree/master/CarND-LaneLines-P1/test_videos_output)  directory

---

### Current Detection Pipeline:
##### 1. Convert image to grayscale:

the 3 color channels is not a variable that is required in the computer vision algorithms that I'll be using, and
as working with one channel data could help making the process easier, just convert the image to grayscale with `cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)`

##### 2.Gaussian blur:

Apply a gaussian filter on the image to help smooth out noise (since we can't assume the image to be that great all the time)
with `cv2.GaussianBlur(img, (k_size, k_size), 0)` where the parameter k_size is the kernel size of the gaussian function and is
something that could be tuned (in my pipeline I am using k_size=5)

##### 3. Edge detection with the Canny filter

Here the edges of all the objects in the photo are found with the `cv2.Canny(img, low_threshold, high_threshold)`:

![canny](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-LaneLines-P1/test_images_output/solidWhiteCurve_edges.jpg)

#### 4. Masking

The left and right lane lines are within a triangle/polygon with the base as the bottom of the image and the apex around the middle--
we only need to focus on this area so masked the rest of the image out and only further process the area of interest with `cv2.fillPoly(imgCopy, vertices, ignore_mask_color)`

Note: this can only be done _after_ canny edge detection otherwise the edge detector will pick up the mask edges

![masked_img](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-LaneLines-P1/test_images_output/solidWhiteCurve_masked.jpg)

##### Hough transform

Apply the hough transform algorithm to detect lines. However, to get 2 long lines corresponding to the lane lines, I combined the results
from the basic line detection hough transform.

Process as follows:
1. find the slope `(y2 - y1)/ (x2 -x1)` of each set of lines found with the hough algorithm. If it's +ve, put it in the right lane line array, if it's -ve, the left lane line array. The is also a 'min slope'
threshold, so lines that are too horizontal is not used (these will be coming from lines that are outside our lane or artifacts)

2. use `np.polyfit` on the left lines x and y values, and on the right lines x and y values to find the equation for the 2 lines.

3. set y endpoint of both lines to be on the bottom of the image, and find the corresponding x values of these points from the equation

4. set the x, y endpoint of the lines around the middle of the image by
using the value of y where it is the smallest in the set of x, y pairs from hough. Then readjust one of the x, y pairs so that the y values of the right and left lines near the center of the image are equal (since the lane lines should converge towards a point in the horizon)

The result from this is then superimposed on the original image. (example below)

##### Final image with lines superimposed:

![final](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-LaneLines-P1/test_images_output/solidWhiteCurve_final.jpg)

### Issues with current pipeline
- curves rather than straight lines. At the moment, the linear interpolation assumes that the road is fairly straight and
the lane lines can be modelled as a linear equation. However, as can be seen in the challenge video, on bendy roads the lines
curve and currently the pipeline doesn't handle this well at all
- jitter between frame to frame. Ideally, would want to feed in the previous positions of x1, y1, x2, y2 of the previous line fit to
the next one so that the transition is smoother
- lines that bend round towards the horizon-- misclassifying some line segments that are actually the right lane line as the left
- very faint/ sparse lane lines-- at the moment the various parameters are tuned to give good results on reasonably clear lane lines. in cases
where the lines are quite faded these might need tweaking, however, it
would also mean that there will be a lot more external noise that creeps in...


### Improvements to investigate
- use a more generalised form of the hough transform that works with curves, (todo:look up GeneralizedHough Transform)
- model the curves as a set of very short lines
- modify the pipeline/ video processing pipeline so you can feed in params from 1 iteration to the next, ie if in frame 1, the bottom x value is at _x<sub>a</sub>_, then at frame 2 the bottom x value shouldn't be more than &#177; &#916;&#949; from _x<sub>a</sub>_ (where &#949; is small)
