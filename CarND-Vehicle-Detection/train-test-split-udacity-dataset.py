import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import glob

udacity_img = glob.glob('udacity_driving_datasets/*.jpg')
udacity_img = [x.split('/')[1] for x in udacity_img]
udacity_img = shuffle(udacity_img)
train_imsg = udacity_img[:int(len(udacity_img) * 0.8)]
val_imgs = udacity_img[int(len(udacity_img) * 0.8):]
labels = pd.read_csv('udacity_driving_datasets/labels.csv')

df_train = labels[labels['frame'].isin(train_imsg)]
df_val = labels[labels['frame'].isin(val_imgs)]
df_train.to_csv('udacity_driving_datasets/train_labels.csv', index=False)
df_val.to_csv('udacity_driving_datasets/val_labels.csv', index=False)