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
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

def logger(epoch, logs):
    if epoch %2== 0:
        print(epoch, logs)
logging_callback = LambdaCallback(on_epoch_end=logger)

# get list of images to use
data_file_paths = ['data2/', 'data_sample/','data_reverse/']
samples = []
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
print(train_samples[:3])

# using the model structure as per nvidia's end to end self driving car paper, with a cropping
# layer for removing the unecessary bits (sky etc)
#  and a normalisation layer
def nvidia_model(input_shape):
    inputs = Input(shape=input_shape)
    cropped = Cropping2D(cropping=((70, 25), (0, 0)))(inputs)
    processed = Lambda(lambda x: (x /255. - 0.5))(cropped)

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

    fcblock = Dense(100, activation='relu', name='fc1')(flat1)
    fcblock = Dense(50, activation='relu', name='fc2')(fcblock)
    fcblock = Dense(10, activation='relu' , name='fc3')(fcblock)

    predictions = Dense(1, name='final')(fcblock)
    model = Model(inputs=inputs, outputs=predictions)
    return model

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

                if (is_validation_generator == False and batch_sample[1].split('/')[0] != 'data_recovery') :
                    left_image = cv2.imread(batch_sample[1])
                    right_image = cv2.imread(batch_sample[2])

                    # images.append(cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB))
                    # angles.append(steering_angle + 0.01 * steering_angle)
                    #
                    # images.append(cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB))
                    # angles.append(steering_angle - 0.005 * steering_angle)


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

K.clear_session()
train_batch_size = 8
val_batch_size = 32
epochs = 30
model = nvidia_model((160, 320, 3))
train_generator = img_generator(train_samples, batch_size=train_batch_size)
validation_generator = img_generator(validation_samples, batch_size=val_batch_size, is_validation_generator = True)

model.compile(loss='mse', optimizer='adam')


model.fit_generator(train_generator, steps_per_epoch=len(train_samples)//train_batch_size, validation_data=validation_generator,
            validation_steps=len(validation_samples)/val_batch_size, epochs=epochs, shuffle=True,
                   callbacks=[logging_callback, ModelCheckpoint('./models/nvidia_generator5.h5', save_best_only=True),
                   TensorBoard(log_dir='./logs/nvidia/generator5' )])
