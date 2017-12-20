import keras
import numpy as np
import pandas as pd
import cv2
import csv

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
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2


K.clear_session()
batch_size = 16
epochs = 5

# keras callbacks
def logger(epoch, logs):
    if epoch %2== 0:
        print(epoch, logs)
logging_callback = LambdaCallback(
    on_epoch_end=logger)

# from https://github.com/keras-team/keras/blob/bc285462ad8ec9b8bc00bd6e09f9bcd9ae3d84a2/examples/cifar10_resnet.py
"""
ResNet v1
[a] Deep Residual Learning for Image Recognition
https://arxiv.org/pdf/1512.03385.pdf
ResNet v2
[b] Identity Mappings in Deep Residual Networks
https://arxiv.org/pdf/1603.05027.pdf
"""


num_classes = 1

n = 3

# Model version
# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
version = 1

# Computed depth from supplied model parameter n
depth = n * 6 + 2

# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, version)


# # Input image dimensions.
# input_shape = x_train.shape[1:]

def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def resnet_block(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        conv_first (bool): conv-bn-activation (True) or
            activation-bn-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    if conv_first:
        x = Conv2D(num_filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-4))(inputs)
        x = BatchNormalization()(x)
        if activation:
            x = Activation(activation)(x)
        return x
    x = BatchNormalization()(inputs)
    if activation:
        x = Activation('relu')(x)
    x = Conv2D(num_filters,
               kernel_size=kernel_size,
               strides=strides,
               padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=l2(1e-4))(x)
    return x


def resnet_v1(input_shape, depth, num_classes=10):
    """ResNet Version 1 Model builder [a]
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    The number of filters doubles when the feature maps size
    is halved.
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    inputs = Input(shape=input_shape)
    num_filters = 16
    num_sub_blocks = int((depth - 2) / 6)

    x = resnet_block(inputs=inputs)
    # Instantiate convolutional base (stack of blocks).
    for i in range(3):
        for j in range(num_sub_blocks):
            strides = 1
            is_first_layer_but_not_first_block = j == 0 and i > 0
            if is_first_layer_but_not_first_block:
                strides = 2
            y = resnet_block(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_block(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if is_first_layer_but_not_first_block:
                x = resnet_block(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters = 2 * num_filters

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v2(input_shape, depth, num_classes=10):
    """ResNet Version 2 Model builder [b]
    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    Features maps sizes: 16(input), 64(1st sub_block), 128(2nd), 256(3rd)
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    inputs = Input(shape=input_shape)
    num_filters_in = 16
    num_filters_out = 64
    filter_multiplier = 4
    num_sub_blocks = int((depth - 2) / 9)

    # v2 performs Conv2D on input w/o BN-ReLU
    x = Conv2D(num_filters_in,
               kernel_size=3,
               padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=l2(1e-4))(inputs)

    # Instantiate convolutional base (stack of blocks).
    for i in range(3):
        if i > 0:
            filter_multiplier = 2
        num_filters_out = num_filters_in * filter_multiplier

        for j in range(num_sub_blocks):
            strides = 1
            is_first_layer_but_not_first_block = j == 0 and i > 0
            if is_first_layer_but_not_first_block:
                strides = 2
            y = resnet_block(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             conv_first=False)
            y = resnet_block(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_block(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if j == 0:
                x = Conv2D(num_filters_out,
                           kernel_size=1,
                           strides=strides,
                           padding='same',
                           kernel_initializer='he_normal',
                           kernel_regularizer=l2(1e-4))(x)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def resnet_custom(input_shape, depth, num_classes=1):
    inputs = Input(shape=input_shape)
    cropped = Cropping2D(cropping=((70, 25), (0, 0)))(inputs)
    processed = Lambda(lambda x: (x /255. - 0.5))(cropped)
    resized = Lambda(lambda x: K.tf.image.resize_images(x, (224, 224)))(processed)

    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_sub_blocks = int((depth - 2) / 6)

    x = resnet_block(inputs=resized)
    # Instantiate convolutional base (stack of blocks).
    for i in range(3):
        for j in range(num_sub_blocks):
            strides = 1
            is_first_layer_but_not_first_block = j == 0 and i > 0
            if is_first_layer_but_not_first_block:
                strides = 2
            y = resnet_block(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_block(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if is_first_layer_but_not_first_block:
                x = resnet_block(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters = 2 * num_filters

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


data_file_paths = ['data_sample/', 'data_recovery/', 'data_reverse/']
images = []
steering = []
# normal center cam + augmentation
lines = []
with open(data_file_paths[0] + '/driving_log.csv') as f:
    reader = csv.reader(f)
    for line in reader:
        lineout = [x.replace('/Users/wwymak/simulator_files', '').replace(' ', '') for x in line]
        lines.append(lineout)

images_set1 = []
steering_set1 = []
for line in lines[1:]:
    img_path = line[0]
    image = cv2.imread(data_file_paths[0] + img_path)
    images_set1.append(image)
    steering_set1.append(float(line[3])* (1 + np.random.normal(0, 0.0005)))

augmented_img = []
augmented_measurements = []

for img, measurement in zip(images_set1, steering_set1):
    augmented_img.append(img)
    augmented_measurements.append(float(measurement))
    augmented_img.append(np.fliplr(img))
    augmented_measurements.append(float(measurement) * -1.0)


images += augmented_img
steering += augmented_measurements


images_set2 = []
steering_set2 = []


for line in lines[1:]:
    left_img_path = line[1]
    right_img_path = line[2]
    steering_angle = float(line[3])

    image_left = cv2.imread(data_file_paths[0] + left_img_path)
    image_right = cv2.imread(data_file_paths[0] + right_img_path)


    images_set2.append(image_left)
    steering_set2.append(steering_angle + np.random.normal(0.01, 0.0005) * steering_angle )
    images_set2.append(image_right)
    steering_set2.append(steering_angle - np.random.normal(0.01, 0.0005) * steering_angle))

images += images_set2
steering += steering_set2

lines = []
with open(data_file_paths[1] + '/driving_log.csv') as f:
    reader = csv.reader(f)
    for line in reader:
        lineout = [x.replace('/Users/wwymak/simulator_files2', '').replace(' ', '') for x in line]
        lines.append(lineout)

images_set3 = []
steering_set3 = []
for line in lines[1:]:
    img_path = line[0]
    image = cv2.imread(data_file_paths[1] + img_path)
    images_set3.append(image)
    steering_set3.append(line[3])

# images += images_set3
# steering += steering_set3


X_train = np.array(images)
y_train = np.array(steering)
model = resnet_custom(input_shape=X_train[0].shape, depth=depth)

model.compile(loss='mse', optimizer='adam')


my_resnet_hist = model.fit(X_train, y_train, batch_size=batch_size,
                    epochs=epochs,validation_split=0.2,
                      verbose=0,shuffle=True,
                    callbacks=[logging_callback, ModelCheckpoint('./models/my_resnet_sampledata_no_recovery.h5', save_best_only=True),TensorBoard(log_dir='./logs/resenet/no_recovery' )])
