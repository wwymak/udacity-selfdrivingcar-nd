### Traffic Sign Recognition

---
This goal of this project is to build, train and test neural network architectures to classify traffic signs. The dataset used is the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)

The full code, including dataset preprocessing, the networks, training results etc are in this [noteobok](https://github.com/wwymak/udacity-selfdrivingcar-nd/tree/master/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

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

Since the class weights can be easily calculated, I used this for a first attempt at training two networks (since if this gives a really good result then there may be no real need to do very fancy data augmentation)

### Data preprocessing

As some training images are quite blurry, I used a histogramEqualisation function from opencv to enhance the contrast. The images are first converted to [YUV colorspace](https://en.wikipedia.org/wiki/YUV) and the Y channel was the one that was used in equalisation (so brightness was normalised). Then, the images are normalised by the sample mean.

The model was getting a fairly good validation accuracy with this preprocessing. However, I am hoping to see a even better accuracy (and a model that can generalise well) with data augmentation (e.g. rotating images, shear, shifts in x and y direction). The image augementaion is handled by using the keras `ImageDataGenerator` which has options for rotation, shift, etc. It also works as a generator for the fit function so even if I decide to use the process on a larger dataset I can do so without running into memory issues.

### Model Architectures

I experimented with three different models:
1. leNet based, with fewer layers, mainly to check that there isn't a mistake in my data preprocessing.
2. vgg based -- I copied the structure of the first 2 conv blocks of the vgg network, with minor adjustments to the filter params. I didn't put in the 3rd conv block since the starting images are only 32x32 and any further scaling down will mean there's hardly any pixels left. The validation accuracy of this architecture is > 96% with image augementation. Vgg was chosen as the base as it had a really good result in imagenet and is straighforward to implement. While traffic signs are not part of imagenet the nature of the photos are very similar
3. multiscale CNN  -- in testing dataset I achieved accuracy of 99%, and and accuracy of 99.6% on the validation set.


Layers in the vgg like model:


| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x32				    |
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x32				    |
| Fully connected		| outputs 1024  									|
| RELU					|												|
| Dropout				| dropout layer  with keep proba=0.25           |
| Fully connected		| outputs 512  									|
| RELU					|												|
| Classification layer	| outputs 43 (num classes)	                    |

##### hyperparameters:
Weights: I used [Xavier initialisation](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) which takes out some of the harder guesswork in hyperparmeters.

Bias: I used the usual zeros initialisation.

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
* training set accuracy of 99.
* validation set accuracy of 99.6%
* test set accuracy of 99%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Stop Sign      		| Stop sign   									|
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .60         			| Stop sign   									|
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ...

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
