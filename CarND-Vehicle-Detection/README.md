## Vehicle Detection Project

The aims of this project is to detect vehicles from a movie shot from the viewpoint of a driver, which is crucial for a self
driving car to be able to handle road situations. The main task is therefore object detection, which has been a long running computer
vision research topic.

In the project rubric for udacity, it calls for using Histogram of Oriented Gradients (HOG) method to extract features and
run a classifier for object detection. However, this is a rather old method that newer deep learning based models has mostly replaced.
Therefore, to get to the end goal of annotating a video stream, I used a neural network based model.

There are 3 parts to the project-- an initial exploration of the traditional HOG extraction so I can gain an understanding of the technique, an experiment with a fine tuned mobilenet using the sliding window method, and utilising one of the newest object
detection models-- single shot detection

### Histogram of Oriented Gradients (HOG)


### Fine tuning Mobilenet with sliding sliding windows

The output video performance is more or less acceptable, detecting the 2 main cars closest to the camera. However, there is still
a lot of jitter and a few false positives, which could potentially be resolved by experimenting with more window sizes and better/
more training data. However, the main issue with this approach is the time taken to process each image. Using python's `time` module, the network takes around 3s per image, which is nowhere near sufficient to run this in real time. (some of the earliest neural network based object detection used this approach, with more or less the same problem.)

The output video of this experiment can be downloaded from [here](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Vehicle-Detection/outvideo_mobilenet_full.mp4)

### Single Shot MultiBox Detector (SSD)

A fairly recent CNN based object detector is the single shot multiBox detector (ref [here](https://arxiv.org/abs/1512.02325)). It is
one of the best performing models in terms of speed and accuracy, (in the paper they quoted a value of 59 frames a second, with 79% mean average precision on the object detection/classification task). I decided that this should be a very good method for the
vehicle detection task for this project, as it would enable an accurate real time car detection.

The network presented in the paper is trained on (and evaluated against) the PascalVOC[http://host.robots.ox.ac.uk/pascal/VOC/]
and [COCO](http://cocodataset.org/) datasets. The Pascal dataset has 20 classes (cars among them)

The original implementation of [SSD](https://github.com/weiliu89/caffe/tree/ssd) is in Caffe, however, as I am more familiar with Keras, I used the Keras implementation of
SSD from  https://github.com/pierluigiferrari/ssd_keras instead. There are a few pretrained networks available from ssd_keras, including ssd300 (trained on the Pascal VOC dataset)-- based on the original Caffe implementation and follows the paper, as well as an example of a smaller custom network. SSD300 is 

For my own custom model, I used the [udacity annotated driving dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) as the training dataset. The dataset has around 20K images in total, matching
the size of the Pascal dataset, but has more relevant classes (although the Pascal dataset has cars too). The KITTI/udacity datasets provided
for the project does not fit with my requirements for model training as that is purely for differentiating betweeen cars and not cars,
rather than a set of images with annotated bounding boxes of where the cars are

#### Results

The best video is from the pretrained SSD300 network [here](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Vehicle-Detection/ssd_300_v1.mp4). While it does not detect the small vehicles in the distance, it does not throw up false positives


#### Experimenting with different SSDs:




The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Further work
I would like to tune the video processing pipeline further to reduce the 'jitter', e.g. averaging over frames.

To further my understanding of the SSD architecture, I would also like to spend some time reimplementing from scratch in Keras
a SSD network, as well as explore other recent developments, e.g. the YOLO9000 architecture. Also interesting to explore
would be semantic segmentation (a good overview of this [here](http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review)), detecting the actual cars's shape as opposed to a bounding box, and using this to assign pixels in a video frame to different
objects.
