# imports packages needed in the script
import keras
import numpy as np
import pandas as pd
import cv2
import csv
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential, Model,load_model
from keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, Input, Lambda, Cropping2D
from keras import utils
from keras.callbacks import Callback, LambdaCallback, EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler, ReduceLROnPlateau
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras import backend as K
from keras.preprocessing import image
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

# logging callback for keras-- outputs validation and training losses per 2 episodes. This is mainly for
# if using ipython notebooks since the verbose mode tends to crash the browser
def logger(epoch, logs):
    if epoch %2== 0:
        print(epoch, logs)
logging_callback = LambdaCallback(on_epoch_end=logger)

# get list of images to use-- each folder is one training set

# data_reverse is driving in the oppositie direction, data2 is my own data, and data_sample is the training data provided
data_file_paths = ['data2/', 'data_sample/','data_reverse/']
samples = []
# read in all the image paths, and use sklearn to split them into training and validation data (using the default shuffle== True)
for folder in data_file_paths:
    temp = []
    with open(folder + 'driving_log.csv') as f:
        reader = csv.reader(f)
        for line in reader:
            lineout = [a.split('/')[-1] for a in line]
            lineout2 = [(folder + 'IMG/' + x) if x.endswith('.jpg') else x for x in lineout]
            temp.append(lineout2)
    samples += temp[1:]
train_samples, validation_samples = train_test_split(samples, test_size=0.1)


# using the model structure as per nvidia's end to end self driving car paper, with a cropping
# layer for removing the unecessary bits (sky etc)
#  and a normalisation layer to limit the inputs to values that the activation functions work better on (mean 0, max 0.5, range 1)
def nvidia_model(input_shape):
    """
    model structure as follows: using the same structure as the end to end learning for
    self driving cars by Nvidia
    _________________________________________________________________
    Layer (type)                 Output Shape              Params
    =================================================================
    input_1 (InputLayer)         (None, 160, 320, 3)       0
    _________________________________________________________________
    cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0
    _________________________________________________________________
    lambda_1 (Lambda)            (None, 65, 320, 3)        0
    _________________________________________________________________
    set1_conv1 (Conv2D)          (None, 33, 160, 24)       1824
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 33, 160, 24)       96
    _________________________________________________________________
    set2_conv1 (Conv2D)          (None, 17, 80, 36)        21636
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 17, 80, 36)        144
    _________________________________________________________________
    set3_conv1 (Conv2D)          (None, 9, 40, 48)         43248
    _________________________________________________________________
    batch_normalization_3 (Batch (None, 9, 40, 48)         192
    _________________________________________________________________
    set4_conv1 (Conv2D)          (None, 9, 40, 64)         27712
    _________________________________________________________________
    batch_normalization_4 (Batch (None, 9, 40, 64)         256
    _________________________________________________________________
    set5_conv1 (Conv2D)          (None, 9, 40, 64)         36928
    _________________________________________________________________
    batch_normalization_5 (Batch (None, 9, 40, 64)         256
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 23040)             0
    _________________________________________________________________
    fc1 (Dense)                  (None, 100)               2304100
    _________________________________________________________________
    fc2 (Dense)                  (None, 50)                5050
    _________________________________________________________________
    fc3 (Dense)                  (None, 10)                510
    _________________________________________________________________
    final (Dense)                (None, 1)                 11
    =================================================================
    Total params: 2,441,963
    Trainable params: 2,441,491
    Non-trainable params: 472
    _________________________________________________________________

    """

    inputs = Input(shape=input_shape)
    cropped = Cropping2D(cropping=((70, 25), (0, 0)))(inputs)
    processed = Lambda(lambda x: (x /255. - 0.5))(cropped)
    # 5 convolution blocks, with Relu activation and batch normalisation
    block1 = Conv2D(24, kernel_size=5,strides=(2,2), activation='relu', padding='same', name='set1_conv1')(processed)
    block1 = BatchNormalization()(block1)
    block2 = Conv2D(36, kernel_size=5,strides=(2,2), activation='relu', padding='same', name='set2_conv1')(block1)
    block2 = BatchNormalization()(block2)
    block3 = Conv2D(48, kernel_size=5,strides=(2,2), activation='relu', padding='same', name='set3_conv1')(block2)
    block3 = BatchNormalization()(block3)
    block4 = Conv2D(64, kernel_size=3, activation='relu', padding='same', name='set4_conv1')(block3)
    block4 = BatchNormalization()(block4)
    block5 = Conv2D(64, kernel_size=3, activation='relu', padding='same', name='set5_conv1')(block4)
    block5 = BatchNormalization()(block5)

    flat1 = Flatten()(block5)
    # 3 fully connected blocks
    fcblock = Dense(100, activation='relu', name='fc1')(flat1)
    fcblock = Dense(50, activation='relu', name='fc2')(fcblock)
    fcblock = Dense(10, activation='relu' , name='fc3')(fcblock)
    # 1 output : this is the steering angle
    predictions = Dense(1, name='final')(fcblock)
    model = Model(inputs=inputs, outputs=predictions)
    return model

# generator for producing training/validation data in batches so there is no need to hold all of
#  the training data in memory-- given the sample size, the machine would actually run out of memory if I use
# all three training datasets (sample driving data, my driving data, and the reverse driving)
def img_generator(samples, batch_size=32, is_validation_generator = False):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        # shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            for batch_sample in batch_samples:
                steering_angle = float(batch_sample[3])

                center_image = cv2.imread(batch_sample[0])
                images.append(cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB))
                angles.append(steering_angle)

                images.append(np.fliplr(center_image))
                angles.append(steering_angle * - 1.0)

                # the validation data shouldn't have the extra left/ right or the reversed images as
                # at testing time only data from central camera is used
                if (is_validation_generator == False and batch_sample[1].split('/')[0] != 'data_recovery') :
                    left_image = cv2.imread(batch_sample[1])
                    right_image = cv2.imread(batch_sample[2])


                    if batch_sample[1].split('/')[0] == 'data_reverse':
                        images.append(cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB))
                        angles.append(steering_angle + 0.005 * steering_angle)

                        images.append(cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB))
                        angles.append(steering_angle - 0.008 * steering_angle)

                    else :
                        images.append(cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB))
                        angles.append(steering_angle + 0.01 * steering_angle)

                        images.append(cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB))
                        angles.append(steering_angle - 0.005 * steering_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# training params. the K.clear_session() is used to remove any stale models from the GPU so it doesn't run out of memory
K.clear_session()
train_batch_size = 8 #use a smaller training batch size since 1 input image turns into 4 by the generator
val_batch_size = 32
epochs = 30
# defined the model
model = nvidia_model((160, 320, 3))
# and the source data generators
train_generator = img_generator(train_samples, batch_size=train_batch_size)
validation_generator = img_generator(validation_samples, batch_size=val_batch_size, is_validation_generator = True)

# set up the model to use mean squared error as loss function and Adam optimisation
model.compile(loss='mse', optimizer='adam')

# train model, fit to data
model.fit_generator(train_generator, steps_per_epoch=len(train_samples)//train_batch_size, validation_data=validation_generator,
            validation_steps=len(validation_samples)/val_batch_size, epochs=epochs, shuffle=True,
                   callbacks=[logging_callback, ModelCheckpoint('./models/nvidia_generator5.h5', save_best_only=True),
                   TensorBoard(log_dir='./logs/nvidia/generator5' )])
