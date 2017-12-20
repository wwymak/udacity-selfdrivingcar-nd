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


K.clear_session()
batch_size = 16
epochs = 5

# keras callbacks
def logger(epoch, logs):
    if epoch %2== 0:
        print(epoch, logs)
logging_callback = LambdaCallback(on_epoch_end=logger)

# get list of images to use
data_file_paths = ['data_sample/', 'data2/', 'data_recovery/', 'data_reverse/']
samples = []
for folder in data_file_paths:
    temp = []
    with open(folder + 'driving_log.csv') as f:
        reader = csv.reader(f)
        for line in reader:
            lineout = [a.split('/')[len(a.split('/'))-1] for a in line]
            lineout2 = [(folder + 'IMG/' + x) if x.endswith('.jpg') else x for x in lineout]
            temp.append(lineout2)
    samples += temp[1:]

train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print(train_samples[:3])

def generator(samples, batch_size=8):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            for batch_sample in batch_samples:
                center_image = cv2.imread(batch_sample[0])
                left_image = cv2.imread(batch_sample[1])
                right_image = cv2.imread(batch_sample[2])

                center_angle = float(batch_sample[3])

                images.append(cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB))
                angles.append(center_angle)

                images.append(cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB))
                angles.append(center_angle * (1 + np.random.normal(0.01, 0.005)))

                images.append(cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB))
                angles.append(center_angle * (1 - np.random.normal(0.01, 0.005)))

                images.append(np.fliplr(center_image))
                angles.append(float(center_angle) * - 1.0)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield X_train, y_train

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

def vgglike(input_shape, ksize=(3,3), dropout=0.25):
    inputs = Input(shape=input_shape)
    cropped = Cropping2D(cropping=((70, 25), (0, 0)))(inputs)
    processed = Lambda(lambda x: (x /255. - 0.5))(cropped)

    block1 = Conv2D(32, kernel_size=ksize, activation='relu', padding='same', name='set1_conv1')(processed)
    block1 = BatchNormalization()(block1)
    block1 = Conv2D(32, kernel_size=ksize, activation='relu', padding='same', name='set1_conv2')(block1)
    block1 = BatchNormalization()(block1)
    block1 = MaxPooling2D(pool_size=(2, 2),strides=(2, 2), name='set1_pool')(block1)

    block2 = Conv2D(64, kernel_size=ksize, activation='relu', padding='same', name='set2_conv1')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Conv2D(64, kernel_size=ksize, activation='relu', padding='same', name='set2_conv2')(block2)
    block2 = BatchNormalization()(block2)
    block2 = Conv2D(64, kernel_size=ksize, activation='relu', padding='same', name='set2_conv3')(block2)
    block2 = BatchNormalization()(block2)
    block2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),name='set2_pool')(block2)
    output2 = Flatten()(block2)

    fcblock = Dense(1024, activation='relu', name='fc1')(output2)
    fcblock = BatchNormalization()(fcblock)
    fcblock = Dropout(dropout)(fcblock)
    fcblock = Dense(512, activation='relu' , name='fc2')(fcblock)
    fcblock = BatchNormalization()(fcblock)
    fcblock = Dropout(dropout)(fcblock)

    predictions = Dense(1, name='final')(fcblock)
    model = Model(inputs=inputs, outputs=predictions)

train_generator = generator(train_samples, batch_size=8)
validation_generator = generator(validation_samples, batch_size=8)

model = nvidia_model((160, 320, 3))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch=len(train_samples), validation_data=validation_generator,
            validation_steps=len(validation_samples), epochs=epochs, shuffle=True,
                   callbacks=[logging_callback, ModelCheckpoint('./models/nvidia_alldata.h5', save_best_only=True),TensorBoard(log_dir='./logs/nvidia/alldata' )])
