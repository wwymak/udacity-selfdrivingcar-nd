## Vehicle Detection Project

The aims of this project is to detect vehicles from a movie shot from the viewpoint of a driver, which is crucial for a self
driving car to be able to handle road situations. The main task is therefore object detection, which has been a long running computer
vision research topic.

In the project rubric for udacity, it calls for using Histogram of Oriented Gradients (HOG) method to extract features and
run a classifier for object detection. However, this is a rather old method that newer deep learning/CNN based models has mostly replaced.
([this paper](http://arxiv.org/abs/1704.05519) is a good reference of what is the state of art computer vision techniques for self driving cars, and HOG has not been 'state of the art' anymore for a while).
Therefore, to get to the end goal of annotating a video stream, I used a neural network based model.

There are 3 parts to the project-- an initial exploration of the traditional HOG extraction so I can gain an understanding of the technique, an experiment with a fine tuned mobilenet using the sliding window method, and utilising one of the newest object
detection models-- Single Shot Multibox Detection. (SSD) The focus of the discussion will be on the SSD (and also a bit on the CNN with sliding windows.) 


### Fine tuning Mobilenet, vehicle detection in images with sliding windows

The output video performance is more or less acceptable, detecting the 2 main cars closest to the camera. However, there is still
a lot of jitter and a few false positives, which could potentially be resolved by experimenting with more window sizes and better/
more training data. However, the main issue with this approach is the time taken to process each image. Using python's `time` module, the network takes around 3s per image, which is nowhere near sufficient to run this in real time. (some of the earliest neural network based object detection used this approach, with more or less the same problem.)

The output video of this experiment is [here](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Vehicle-Detection/outvideo_mobilenet_full.mp4)

### Single Shot MultiBox Detector (SSD)

A fairly recent CNN based object detector is the single shot multiBox detector (ref [here](https://arxiv.org/abs/1512.02325)). It is
one of the best performing models in terms of speed and accuracy, (in the paper they quoted a value of 59 frames a second, with 79% mean average precision on the object detection/classification task). I decided that this should be a very good method for the
vehicle detection task for this project, as it would enable an accurate real time car detection.

The basics  of the network is as follows:

![](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Vehicle-Detection/examples/SSD_architecture.png).

The base model of the VGG16 is used to extract features

The network presented in the paper is trained on (and evaluated against) the PascalVOC[http://host.robots.ox.ac.uk/pascal/VOC/]
and [COCO](http://cocodataset.org/) datasets. The Pascal dataset has 20 classes (cars among them)

The original implementation of [SSD](https://github.com/weiliu89/caffe/tree/ssd) is in Caffe, however, as I am more familiar with Keras, I used the Keras implementation of
SSD from  https://github.com/pierluigiferrari/ssd_keras instead. There are a few pretrained networks available from ssd_keras, including ssd300 (trained on the Pascal VOC dataset)-- based on the original Caffe implementation and follows the paper, as well as an example of a smaller custom network.

I tested 2 models on the vehicle detection problem-- one is a smaller network I developed with a mobilenet backend, and the
4 classification + 4 bounding box detection layers branching off from the last 4 conv blocks, and also the full SSD300 network
(using the pretrained weights)

The training script for the mobilenet ssd is [here](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Vehicle-Detection/mobilenet-ssd-training.py) and the
prediction and movie creation task on this network is [here](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Vehicle-Detection/mobilenet-ssd-predict.ipynb)

#### Mobilenet SSD
As a learning task, I also constructed a SSD network with a mobilenet backend instead of the VGG16 backend.  

The network architecture is shown [here](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Vehicle-Detection/mobilenet-architecture.md).

For my own custom model, I used the traffic [dataset](https://drive.google.com/file/d/0B0WbA4IemlxlT1IzQ0U1S2xHYVU/view?usp=sharing) provided by ssd_keras (this is based on the [udacity annotated driving dataset](https://github.com/udacity/self-driving-car/tree/master/annotations), but with images scaled down to 300 x 480, which is
much more manageable for a neural network ) as the training dataset. The dataset has around 20K images in total, matching
the size of the Pascal dataset, but has more relevant classes (although the Pascal dataset has cars too).
In the current implementation, I trained the model from scratch over 70 epochs. As mentioned in the paper, data augmentation is
very important to get good accuracy in training,

**Prediction**

The code in [mobilenet-ssd-predict.ipynb](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Vehicle-Detection/mobilenet-ssd-predict.ipynb) shows how to use the mobilenet SSD to detect vehicles, as well as the movie processing pipeline. There are more details in the notebook, but the main observations are as follows:

In terms of predicting on the training/validation data from the udacity traffic dataset:


#### Results of SSD

The best video is from the pretrained SSD300 network [here](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Vehicle-Detection/ssd_300_v1.mp4). While it does not detect the small vehicles in the distance/ at the opposite lane, it does not throw up any false positives, with no further processing on video frames needed. This is not perhaps not surprising as the pretrained SSD300 has been trained for a much higher number of steps
compared to my mobilenet SSD, and also has undergone a detailed parameter tuning (e.g. the different box scales).

My own

Performance wise, the movie prediction pipeline takes approx 2-3 mins for 50s of video. The video has a frame rate of 25fps, if we ignore the other non-SSD processing required for each frame(e.g. the image resizes, averaging calculations, etc), the network is
running at 7 frames per second. With further optimsation of the processing pipeline so there is less out of GPU operations
needed, the network is likely to be even faster-- and gets close to the goal of real time object detection.


### Further work
I would like to tune the video processing pipeline further to reduce the 'jitter', e.g. averaging over frames.

To further my understanding of the SSD architecture, I would also like to spend some time reimplementing from scratch in Keras
a SSD network, as well as explore other recent developments, e.g. the YOLO9000 architecture. Also interesting to explore
would be semantic segmentation (a good overview of this [here](http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review)), detecting the actual cars's shape as opposed to a bounding box, and using this to assign pixels in a video frame to different
objects.
