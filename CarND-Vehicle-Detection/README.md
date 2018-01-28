# Vehicle Detection Project

The aims of this project is to detect vehicles from a movie shot from the viewpoint of a driver, which is crucial for a self
driving car to be able to handle road situations. The main task is therefore object detection, which has been a long running computer
vision research topic.

In the project rubric for udacity, it calls for using Histogram of Oriented Gradients (HOG) method to extract features and
run a classifier for object detection. However, this is a rather old method that newer deep learning/CNN based models has mostly replaced.
([this paper](http://arxiv.org/abs/1704.05519) is a good reference of what is the state of art computer vision techniques for self driving cars, and HOG has not been 'state of the art' anymore for a while).
Therefore, to get to the end goal of annotating a video stream, I used a neural network based model.

There are 3 parts to the project-- an initial exploration of the traditional HOG extraction so I can gain an understanding of the technique, an experiment with a fine tuned mobilenet using the sliding window method, and utilising one of the newest object
detection models-- Single Shot Multibox Detection. (SSD) The focus of the discussion will be on the SSD (and also a bit on the CNN with sliding windows.). My brief exploration of HOG methods is in [Object-detection1.ipynb](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Vehicle-Detection/Object-detection1.ipynb) (but as it is not well annotated and I didn't go very far down this route it's not very interesting to look at)

---

## Fine tuning Mobilenet, vehicle detection in images with sliding windows

A basic implementation of object detection with CNNs is to split up each image into windows of set sizes, then run a CNN
trained on detecting cars to run over these windows and classify them as containing a car or not. While I know that this
method is going to be really slow, I am curious as to how well it would work, and just how well(or badly) would it cope
with video processing.

I decided to fine tune a pretrianed mobilenet model from the keras library on the cars/not cars dataset provided for the project.
The code for this is at [mobilenet fine tune.ipynb](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Vehicle-Detection/mobilenet%20fine%20tune.ipynb)
After 50 epochs of training (with data augmentation), the network is able to distinguish between cars and not cars with 99% accuracy.


The output video performance is more or less acceptable, detecting the 2 main cars closest to the camera. However, there is still
a lot of jitter and a few false positives, which could potentially be resolved by experimenting with more window sizes and better/
more training data. However, the main issue with this approach is the time taken to process each image. Using python's `time` module, the network takes around 3s per image, which is nowhere near sufficient to run this in real time. (some of the earliest neural network based object detection used this approach, with more or less the same problem.)

The output video of this experiment is [here](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Vehicle-Detection/outvideo_mobilenet_full.mp4)

## Single Shot MultiBox Detector (SSD)

A fairly recent CNN based object detector is the single shot multiBox detector (ref [here](https://arxiv.org/abs/1512.02325)). It is
one of the best performing models in terms of speed and accuracy, (in the paper they quoted a value of 59 frames a second, with 79% mean average precision on the object detection/classification task). I decided that this should be a very good method for the
vehicle detection task for this project, as it would enable an accurate real time car detection.

The structure  of the network is as follows:

![](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Vehicle-Detection/examples/SSD_architecture.png).

The base model of the VGG16 is used to extract features, but rather than passing the features through dense layers, the
SSD models used convolution layers as

The network presented in the paper is trained on (and evaluated against) the [PascalVOC](http://host.robots.ox.ac.uk/pascal/VOC/)
and [COCO](http://cocodataset.org/) datasets. Both the Pascal and COCO datasets has cars among their object detection/classification classes. Therefore, it is entirely possible to use a pretrained network from the paper, filtering on the 'cars' class. However,
the original implementation of [SSD](https://github.com/weiliu89/caffe/tree/ssd) is in Caffe, and as I am more familiar with Keras, I used the Keras implementation of
SSD from  https://github.com/pierluigiferrari/ssd_keras instead. Besides very good documnetation and explanation in the code in ssd_keras about how the SSD is reimplementated in Keras, there is also a lot of useful utility classes and layers that makes constructing my onw network easier.

I tested 2 models on the vehicle detection problem-- one is a smaller network I developed with a mobilenet backend, and the
4 classification + 4 bounding box detection layers branching off from the last 4 conv blocks, and also the full SSD300 network
(using the pretrained weights)

The training script for the mobilenet ssd is [here](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Vehicle-Detection/mobilenet-ssd-training.py) and the
prediction and movie creation task on this network is [here](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Vehicle-Detection/mobilenet-ssd-predict.ipynb)

### Mobilenet SSD
As a learning task, I also constructed a SSD network with a mobilenet backend instead of the VGG16 backend.  

The network architecture is shown [here](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Vehicle-Detection/mobilenet-architecture.md).

It uses [mobilenet](https://arxiv.org/pdf/1704.04861.pdf) as the feature extractor, and the output from the last 4 pointwise conv layers (after relu activation) is fed into the object classifier and the box classifiers for object detection. I chose mobilenet as
a base as mobilenets are optimsed to be more efficient in terms of computing time and is also less memory intense. This is
likely to be useful in a self driving car situation where the detection should run as close to real time as possible and the
model should also not require a lot of computing power to run.

For training, I used the [traffic dataset](https://drive.google.com/file/d/0B0WbA4IemlxlT1IzQ0U1S2xHYVU/view?usp=sharing) provided by ssd_keras (this is based on the [udacity annotated driving dataset](https://github.com/udacity/self-driving-car/tree/master/annotations), but with images scaled down to 300 x 480, which is
much more manageable for a neural network ). The dataset has around 20K images in total, matching
the size of the Pascal dataset, but has more relevant classes (although the Pascal dataset has cars too).
In the current implementation, I trained the model from scratch over 70 epochs. As mentioned in the paper, data augmentation is
very important to get good accuracy in training, and I used translation, horizontal flips, brightness variation and scaling for this.
The code for the training process is [mobilenet-ssd-training.py](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Vehicle-Detection/mobilenet-ssd-training.py)

#### Prediction

The code in [mobilenet-ssd-predict.ipynb](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Vehicle-Detection/mobilenet-ssd-predict.ipynb) shows how to use the mobilenet SSD to detect vehicles, as well as the movie processing pipeline. There are more details in the notebook, but the main observations are as follows:

**Prediction on udacity traffic dataset**

The following are a set of 6 images randomly picked from the udacity traffic dataset. As can be seen, the model did a pretty good job
at detecting the bounding boxes for the cars. So, would this translate to the test images and the video?

![](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Vehicle-Detection/output_images/1478732080090015975_predicted.jpg)   ![](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Vehicle-Detection/output_images/1478895368744352345_predicted.jpg)   

![](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Vehicle-Detection/output_images/1478897820720062731_predicted.jpg)   ![](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Vehicle-Detection/output_images/1478901524392001997_predicted.jpg)   

![](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Vehicle-Detection/output_images/1479498540474511391_predicted.jpg)   ![](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Vehicle-Detection/output_images/1479502464744941068_predicted.jpg)


**Prediction on the test images**

The following is the model on the test images, which are what the video frames look like. The model works fairly well, but
has false positives, detecting cars in the 'shadows' along the left railing. Potentially, this could be because the training dataset
has more images with cars in shadows, or that it has not learnt to deal with the high light/dark contrast very well. In my video pipeline (discussion below), I implemented an 'averaging' filter to reduce the false positives.

![](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Vehicle-Detection/output_images/test1_predicted.jpg)   ![](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Vehicle-Detection/output_images/test2_predicted.jpg)   

![](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Vehicle-Detection/output_images/test3_predicted.jpg)   ![](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Vehicle-Detection/output_images/test4_predicted.jpg)   

![](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Vehicle-Detection/output_images/test5_predicted.jpg)   ![](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Vehicle-Detection/output_images/test6_predicted.jpg)

*my mobilenet ssd model can be downloaded from https://storage.googleapis.com/udacity-sdcnd-misc/CarND-Vehicle-Detection/ssd7_train/ssd7_mobilenet_v1.h5 and the weights from https://storage.googleapis.com/udacity-sdcnd-misc/CarND-Vehicle-Detection/ssd7_train/ssd7_mobilenet_v1_weights.h5*  

#### Video processing

The video processed using the pretrained SSD300 network is [ssd_300_v1.mp4](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Vehicle-Detection/ssd_300_v1.mp4), and
the video processed using my mobilenet SSD network is [ssd_mobilenet_with_averaging.mp4](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Vehicle-Detection/ssd_mobilenet_with_averaging.mp4)

For the pretrained SSD300 network, no further processing beyond passing each image from the video into the network is necessary-- I found that while it does not detect the small vehicles in the distance/ at the opposite lane, it does not throw up any false positives, and also tracks the closest cars from frame to frame very smoothly. This is not perhaps not surprising as the pretrained SSD300 has been trained for a much higher number of steps
compared to my mobilenet SSD, and also has undergone a detailed parameter tuning.

For the mobilenet SSD, I implemented an 'averaging' over 25 frames. In the video processing pipeline, the boxes from the last 25 frames
are added to a list, and a 'heatmap' constructed out of these boxes. Only areas occurs in more than 15 frames out of 25 is assigned to the final box for a car (I used 15 as it seems to have the best tradeoff between removing false positives and ensuring that the detection for the closest cars don't lag)

Performance wise, the movie prediction pipeline takes approx 2-3 mins for 50s of video. The video has a frame rate of 25fps, if we ignore the other non-SSD processing required for each frame(e.g. the image resizes, averaging calculations, etc), the network is
running at 7 frames per second. With further optimsation of the processing pipeline so there is less out of GPU operations
needed, the network is likely to be even faster-- and gets close to the goal of real time object detection.


### Further work

- Do some more experimentation on parameter tuning, such as the different scaling/variances for the anchor boxes, different training
data augmentation techniques, etc.

- To further my understanding of the SSD architecture, I would also like to spend some time reimplementing from scratch in Keras
a SSD network, as well as explore other recent developments, e.g. the YOLO9000 architecture.

- explore semantic segmentation (a good overview of this [here](http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review)), detecting the actual cars's shape as opposed to a bounding box, and using this to assign pixels in a video frame to different objects, such as lane lines, cars, traffic signs etc

<small>Note: the `keras_ssd*.py`, `keras_layer_*.py` files are from the [ssd_keras project](https://github.com/pierluigiferrari/ssd_keras)</small>
