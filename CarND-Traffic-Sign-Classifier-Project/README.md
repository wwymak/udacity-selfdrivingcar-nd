### Traffic Sign Recognition

---
This goal of this project is to build, train and test neural network architectures to classify traffic signs. The dataset used is the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)

The full code, including dataset preprocessing, the networks, training results etc are in this [notebook](https://github.com/wwymak/udacity-selfdrivingcar-nd/tree/master/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classification.ipynb)

*  note: Using keras version 2.1.1 and tensorflow v1.3

Some of the experiments (e.g. variations in image preprocessing, network architecture) are also as ipynb notebooks in that directory. However, these are messier and less well annotated than the main notebook.

The following are some highlights from the project

##### Setup

- The actual [provided dataset](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip) has been preprocessed slightly so it has
all been resized to 32 x 32 x3 images

- Besides the usual numpy, pandas etc packages, I am also using keras with tensorflow backend for the CNN, as it lets me explore more easily without getting hung up on low level tensorflow apis.

##### Data Set Summary

Here are some rescaled example images from the dataset
![example images](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Traffic-Sign-Classifier-Project/examples/train_examples.jpg)

Summary stats of the provided training/validation/test datsets:

- Number of training examples = 34799
- Number of validation examples = 4410
- Number of testing examples = 12630
- Image data shape = [32, 32, 3]
- Number of classes = 43

##### Distribution of classes:

Ideally, the number of training samples for each class should be roughly equal, as otherwise the network might not learn enough
details about minority classes to be able to predict them at all. So as a first check, this is what the class distribution looks like
for the training and validation sets:

![training data distribution](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Traffic-Sign-Classifier-Project/examples/train_distribution.jpg)

![validation data distribution](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Traffic-Sign-Classifier-Project/examples/validation_distribution.jpg)

As can be seen, the number of samples for classes are quite different. Two possible strategies to address this-- make 'fake' images by image
augementaion, e.g. rotation, shifting etc for the classes with small samples, or to use the `class_weight` param in keras fit which takes this into account. (According to [keras docs](https://keras.io/models/sequential/), the class_weight does

>  class_weight: Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). This can be useful to tell the model to "pay more attention" to samples from an under-represented class.

Since the class weights can be easily calculated, I initially plan on using this for a first attempt at training the networks. However, I found that the results was really good without requiring the class weight tweaking, so perhaps there are enough training data that the unbalanced classes does not affect the results overmuch.

### Data preprocessing

As some training images are quite blurry, I used a histogramEqualisation function from opencv to enhance the contrast. The images are first converted to [YUV colorspace](https://en.wikipedia.org/wiki/YUV) and the Y channel was the one that was used in equalisation (so brightness was normalised). Then, the images are normalised by the sample mean.

The model was getting a fairly good validation accuracy with this preprocessing. However, I am hoping to see a even better accuracy (and a model that can generalise well) with data augmentation (e.g. rotating images, shear, shifts in x and y direction). The image augmentaion is handled by using the keras `ImageDataGenerator` which has options for rotation, shift, etc. It also works as a generator for the fit function so even if I decide to use the process on a larger dataset I can do so without running into memory issues.

Here is example of what the image augmentation does on a test image is:

![](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Traffic-Sign-Classifier-Project/examples/data_augment.png)

[This notebook](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Traffic-Sign-Classifier-Project/image_preprocessing.ipynb) compares the results from training with and without the augmentation. While the difference is not huge, and some of the improvements might be due to random chance, it still seems that image augmentation does help in increasing the test accuracy.

### Model Architectures

I experimented with three different models:
1. leNet based, with fewer layers, mainly to check that there isn't a mistake in my data preprocessing.
2. vgg based -- I copied the structure of the first 2 conv blocks of the vgg network, with minor adjustments to the filter params. I didn't put in the 3rd conv block since the starting images are only 32x32 and any further scaling down will mean there's hardly any pixels left. The validation accuracy of this architecture is > 96% with image augementation. Vgg was chosen as the base as it had a really good result in imagenet and is straighforward to implement. While traffic signs are not part of imagenet the nature of the photos are very similar
3. multiscale CNN  -- similar to the architecture above, but with output from the 1st conv block feeding directly into the fully connected layer after the 2nd conv block. The argument behind why this might work better is that both simple and more complex features are used for learning (this is following along the architecture in Yann LeCun's work on the same task http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf ). This model has the best result-- in testing dataset I achieved accuracy of 99%, and and accuracy of 99.6% on the validation set.


Layers in the vgg like model:


| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| BatchNorm             |                                               |
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| BatchNorm             |                                               |
| Max pooling	      	| 2x2 stride,  outputs 16x16x32				    |
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| BatchNorm             |                                               |
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| BatchNorm             |                                               |
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| BatchNorm             |                                               |
| Max pooling	      	| 2x2 stride,  outputs 8x8x32				    |
| Fully connected		| outputs 1024  								|
| RELU					|												|
| Dropout				| dropout layer  with keep proba=0.25           |
| Fully connected		| outputs 512  									|
| RELU					|												|
| Classification layer	| outputs 43 (num classes)	                    |

Layers in the multiscale cnn model:

These are the same as above, but the output from the 1st maxpool layer is also fed directly into the fully connected layer.

##### hyperparameters:
Weights: I used [Xavier initialisation](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) which takes out some of the harder guesswork in hyperparmeters.

Bias: I used the usual zeros initialisation.

Dropout: this was used (with dropout probability of 0.25) to prevent overfitting. The value used gave a very high train and validation accuracy so I didn't further tune it.

Number of epochs: I found that approximately 30-40 epochs on the vgg like architecture yields a good validation accuracy (approx 96-97%), and a test accuracy of ~95%. For the multiscale cnn, I used epochs of up to 80, but used the Modelcheckpoint callback from keras to save the best model for testing.

##### Optimizer

To train the model, the optimizer used was [Adam](https://arxiv.org/pdf/1412.6980.pdf), with an evaluation metric of accuracy. The Adam optimisation technique is fairly good for this problem as it is fairly forgiving in terms of hyperparmeter tuning, (as hyperparameters such as learning rate gets tweaked based on how the gradients etc are changing)

##### Results

The final results were

**Using the deeper vgg like network:**
* training set accuracy of 0.98
* validation set accuracy of 0.97
* test set accuracy of 0.96

**Multiscale cnn**
* training set accuracy of 99.8%
* validation set accuracy of 99.6%
* test set accuracy of 99%

In all of the models, I plotted the train/validation loss and accuracy to
check that the model is not overfitting or underfitting (both the train and validation losses were almost dropping to zero so no issues of those)


##### Testing the multiscale model

8 traffic signs photos from the internet was used in testing the model

Results are :
![test images](https://github.com/wwymak/udacity-selfdrivingcar-nd/blob/master/CarND-Traffic-Sign-Classifier-Project/examples/test_results.png)


The model was able to correctly predict 6 of the 8 traffic signs, which gives an accuracy of 75%. This is quite a bit lower than the test set accuracy. However, the test set has a lot more images so this low accuracy might be due to chance. Also, the test set was already cropped to 32 x 32 and there might be differences in how this is done compared to these 8 images which might affect how accurate the model does (e.g. the images in the dataset seems to be really zoomed into the signs whereas in these 8 test images there is a lot of background)


##### Further work
* can I combine object detection for traffic signs together with a traffic sign classifier?
* visualising layers of the CNN-- beginnings of this in the CNN vis notebook but there are still some bugs to fix since at the moment different layers seem to look the same...
