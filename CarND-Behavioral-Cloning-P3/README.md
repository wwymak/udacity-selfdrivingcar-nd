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
from 160px to 65px, removing 70px from the top (where the sky is) and 25px from the bottom (where the front of the car is). This makes the model quicker to train and require less memory. Also, by cropping out the irrelvant parts of the image, it can prevent the model from training on the wrong features. (e.g. it shouldn't predict steering angles based on whether it can see trees on not.)

#### Data augmentation

- Centre lane driving. Besides my own data for 'good' centre lane driving, I also used the sample data provided so the model can have a great amount of good data to learn from. Also, the two different datasets with different driving style should help
with the model performance, especially as I am not that good at
the actual simulator myself.

- Flip images: to reduce the chance of the model overfitting on left steering or right steering, I also flipped the images and the
steering angles so the car can learn to navigate turns better.


- add in left and right steering images -- these help in predicting when the car should turn left/right according
to the road. In the view from the left camera, the steering angle should be less to the left if the image is taken from the
central camera, and in the view from the right camera, the steering angle should more more to the left if the image is from the
central camera. The extra data from these cameras help in guiding the model to steer the car more towards the centre. This technique
works similar to the 'recovery' images and is a lot easier to obtain. (see comment about recovery images below)


- reverse driving. Similar to the above, I also collected data of the car driving the wrong way round on the track to correct for any left hand turn biases.

- recovery images-- ideally, when the car drive away from the centre of the track, it should learn to go back towards the centre, and to do this, 'recovery' data might be needed, where the model sees images of the sides being too close and it has to fit a higher recovery angle.  However, I tried adding some recovery data to the training set and the car performed worse, not better with the model trained on them. Perhaps, this is due to the quality of the images not being very good (some of the recovery data has a bit of driving towards the sides mixed in). I found the model did well enough without this extra data so rather than spending a lot of time
manually going through the images and throwing out the bad ones, I decided to only use the other training data instead.

The following image shows the above 2 steps:

![image](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Behavioral-Cloning-P3/examples/image_augmentation.png)

## Model Architecture and Training Strategy

#### Architecture

The final model structure (based on that reported from Nvidia, with the extra batchnorm layers added)
is as follows:


| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 160X320x3 RGB image   						|
| Cropping2D            | Crops Image to 65X320x3 RGB                   |
| Lambda                | Normalise image to have mean 0 and max min range of 1 |
| Convolution 5x5     	| 2x2 stride, same padding, outputs 33x160x24 	|
| RELU					|												|
| BatchNorm             |                                               |
| Convolution 5x5     	| 2x2 stride, same padding, outputs 17x80x36 	|
| RELU					|												|
| BatchNorm             |                                               |
| Convolution 5x5     	| 2x2 stride, same padding, outputs 9x40x48 	|
| RELU					|												|
| BatchNorm             |                                               |
| Convolution 5x5     	| 1x1 stride, same padding, outputs 9x40x64  	|
| RELU					|												|
| BatchNorm             |                                               |
| Convolution 5x5     	| 1x1 stride, same padding, outputs 9x40x64  	|
| RELU					|												|
| BatchNorm             |                                               |
| Fully connected		| outputs 100  								    |
| RELU					|												|
| Fully connected		| outputs 50  								    |
| RELU					|												|
| Fully connected		| outputs 10  								    |
| RELU					|												|
| Fully connected		| prediction 1 (regression)  				    |

Previously, I have also tried a version of LeNet and also VGG. However, LeNet model
seems a bit too simple to predict correctly what the angle should be and I was getting
quite high training and validation losses, whereas the VGG model, while it did well, doesn't
seem to offer much advantage in terms of simulator performance over the simpler nvidia model.
I chose the nvidia model as it is much smaller and is faster to train. As both the training and
validation error on this is quite low, it shouldn't be overfitting. (and therefore like
    in the paper, I didn't use any dropout layers)

It would be interesting to see if more complex structures, such as ResNet or InceptionNet would
have better results to the current structure. However, a quick trial of a ResNet indicates that it would take
a long time to train, and given that VGG with its deeper structure seems to offer no extra gains,
I am leaving this experiment on the side for now.

A more interesting avenue to explore is to implement a CNN-LSTM model on the data-- since each subsequent steering
angle is based on the previous few timesteps, it should offer better predictions than just using the image from the
same timestep. Also interesting would be to implement a prediction for the speed and throttle as well as these
are also related to the speed/environment in a real car, e.g. if you are driving round a turn you would slow down.


#### Model parameters

The model used mean squared error as the metric an Adam optimizer (with default params). Learning rate tuning was not required as the Adam optimizer takes care of changing the rate of learning rate as the model trains. I used 30 epochs, with a ModelCheckpoint callback
to save the model weights at each epoch if the training loss is lower. By the end, it wasn't decreasing much so I didn't continue on training.

The following image illustrates the training loss/ validation loss:
![image](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Behavioral-Cloning-P3/examples/nvidia_model_training_log.png)


#### Train/validation data
After reading in all the image paths and corresponding steering angles, I used the `train_test_split` function from sklearn
to shuffle the training data and to hold back 10% as the validation data. (There are enough images from the 3 datasets that 10% validation data should be sufficient and it would benefit the network from having extra training data)

## Usage
### Packages required

The following packages outside the standard numpy/scipy/matplotlib were used: (the version shouldn't matter except for keras and TF)

- Keras v2.1.1
- Tensorflow v.1.3
- opencv v3.2.0
- sklearn v0.18

### Relevant Files
- train_nvidia.py (for training and saving model)
- drive.py (for taking output from the model and sending it to the simulator)
- video.py (converted images from the autonomous session into a video)
- movie1_nvidia.mp4 (video of the car driving with the model)
- models/nvidia_generator5.h5 (trained model)

(the other files can be ignored-- these are rough notes, investigations that are works in progress, etc)

### Running the code
- keras version 2.1.1 and tensorflow v.1.3 was used for training and running the model. If you are running the saved model
with drive.py you will need to check that the keras version in your environment is 2.1.1
- use `python train_nvidia.py` to train the model. This will save the model in the relative path `./models/nvidia_generator5.h5`
(via the keras ModelCheckpoint option). Logs are saved to the `logs` directory so the training can be visualised in Tensorboard.
- also make a directory called data and put the relevant training data in there-- if need be change the `data_file_paths` array in your own code
- launch TensorBoard with `tensorboard --logdir=./logs` to monitor the training process

- use `python drive.py ./nvidia_generator5.h5` to run the car in autonomous mode. Note- there _may_ be potential issues with
running the model in your own machine (rather than the one the model is trained on) as Keras doesn't seem to be able to reload the
Lambda layer correctly when the model is on a different machine (I trained the same model both with and without the Lambda layer on
a cloud GPU instance and when trying to run them on my own local machine, the one without the Lambda layer loaded fine but the one with the layer gives an error-- there have been various discussions/issues on the keras repo around this, e.g. https://github.com/keras-team/keras/issues/6442. If I am productionising the code and model , I would definitely need to resolve this,
but for this project, I train and run drive.py on the same cloud GPU instance, and use that to drive the simulator by forwarding my local port 4567 to the one on the remote machine)



### Improvements

- I would like to get the car to be able to generalise to drive on the second track-- unfortunately this is really hampered by
the lack of training data due to my own not so great driving... might be interesting to explore whether generative adverserial
techniques can be used to generate good driving data from the bad ones but at the moment I don't think I have seen a paper that
 does similar task.

 - Try a Conv-LSTM on the training-- next iteraterion will likely include an investigation into this (due to time constraints I don't have time right at the moment for project deadline...)

 - Make the car drive better by predicting the speed as well as driving angle.

 - layer visualisations of the model-- haven't managed to get the layer visualisations to show meaningful results as yet.
