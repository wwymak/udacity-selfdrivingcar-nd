## Behavioral Cloning Project

This project is to simulate good driving behaviour using deep learning. Data is collected on a simulator track and a
network is trained such that the car can drive round on the track in autonomous mode.

## Data Collection:
The aim of the project is for the car the drive along the center of track1, and thus good centre lane driving data is
needed for the car to emulate. Also, the car has to learn to navigate round bends, and to handle cases where it steers too
much to the left or right.

Ideally, to help the car to generalise better, I would need to drive it through many different tracks. However, I only gather
training data on 1 track of the simulator as I find it very hard to drive the car on track 2, and rather than make the car drive
worse due to bad driving data, I only used data from track1.

#### Image preprocessing
- No grayscaling: in previous project around lane detection, images are grayscaled for further computer vision algorithms to process.
However, no grayscaling is used as the road is not of one consistent texture, and nor are there 'standard' lane lines. I believe that
using all 3 color channels can in fact help the network to learn.


- converting to the right colorscale: in the image processing pipeline, I am using `cv2.imread` to convert images to numpy arrays. This reads in images in the BGR order whereas in the drive.py file, the images are read in using the `Image` function from Pillow, which is in the RGB order.

- image normalisation

- Image cropping-- only the bottom part of the image corresponding to the road is important for the model
to determine the steering angle, so as part of the keras model, there is a cropping layer that crops the image height
from 160px to 65px, removing 70px from the top (where the sky is) and 25px from the bottom (where the front of the car is)

#### Data augmentation

- Centre lane driving. Besides my own data for 'good' centre lane driving, I also used the sample data provided so the model can have a great amount of good data to learn from. Also, the two different datasets with different driving style should help
with the model performance, especially as I am not that good at
the actual simulator myself.

- Flip images: to reduce the chance of the model overfitting on left steering or right steering, I also flipped the images and the
steering angles so the car can learn to navigate turns better.

- reverse driving. Similar to the above, I also collected data of the car driving the wrong way round on the track to correct for any left hand turn biases.

- add in left and right steering images -- these help in predicting when the car should turn left/right according
to the road. In the view from the left camera, the steering angle should be less to the left if the image is taken from the
central camera, and in the view from the right camera, the steering angle should more more to the left if the image is from the
central camera. The extra data from these cameras help in guiding the model to steer the car more towards the centre. This technique
works similar to the 'recovery' images and is a lot easier to obtain. (see comment about recovery images below)

- recovery images-- ideally, when the car drive away from the centre of the track, it should learn to go back towards the centre, and to do this, 'recovery' data might be needed, where the model sees images of the sides being too close and it has to fit a higher recovery angle.  However, I tried adding some recovery data to the training set and the car performed worse, not better with the model trained on them. Perhaps, this is due to the quality of the images not being very good (some of the recovery data has a bit of driving towards the sides mixed in). I found the model did well enough without this extra data so rather than spending a lot of time
manually going through the images and throwing out the bad ones, I decided to only use the other training data instead.

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

### Relevant Files
- train_nvidia.py
- drive.py
- video.py
- 

### Running the code
- use `python train_nvidia.py` to train the model. This will save the model in the relative path `./models/nvidia_generator5.h5`
(via the keras ModelCheckpoint option). Logs are saved to the ./logs directory so the training can be visualised in Tensorboard.

- use `python drive.py ./models/nvidia_generator5.h5` to run the car in autonomous mode. Note- there _may_ be potential issues with
running the model in your own machine (rather than the one the model is trained on) as Keras doesn't seem to be able to reload the
Lambda layer correctly when the model is on a different machine (I trained the same model both with and without the Lambda layer on
a cloud GPU instance and when trying to run them on my own local machine, the one without the Lambda layer loaded fine but the one with the layer gives an error-- there have been various discussions/issues on the keras repo around this, e.g. https://github.com/keras-team/keras/issues/6442. If I am productionising the code and model , I would definitely need to resolve this,
but for this project, I train and run drive.py on the same cloud GPU instance, and use that to drive the simulator by forwarding my local port 4567 to the one on the remote machine)

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* train_nvidia.py containing the script to create and train the model on the architecture used in the Nvidia paper
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* movie1_nvidia.mp4 is a movie showing the car driving round on the track

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24)

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18).

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used mse as the accurcay param and `model.compile(loss='mse', optimizer='adam')` so the learning rate was not tuned.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that ...

Then I ...

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
