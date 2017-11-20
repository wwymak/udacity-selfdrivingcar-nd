import pickle
import numpy as np
import cv2

training_file = './data/train.p'
validation_file= './data/valid.p'
testing_file = './data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


def histogramEqualisation(img):
    original_type = img.dtype
    if img.dtype != 'uint8':
        img = img.astype('uint8')
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    equalised = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return equalised.astype(original_type)

def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    max_x = np.max(x)
    min_x = np.min(x)
    g = np.vectorize(lambda x: (x- min_x)/ (max_x - min_x))

    normalised = g(x)
    return normalised

def hist(X):
    out = []
    for i in range(len(X)):
        out.append(histogramEqualisation(X[i]))
    return np.array(out)

X_train_histogram =  hist(X_train)
X_valid_histogram = hist(X_valid)
X_test_histogram =  hist(X_test)


def normalX(x):
    max_x = np.max(x)
    min_x = np.min(x)
    out = []
    for i in range(len(x)):
        out.append((x- min_x)/ (max_x - min_x))
    return np.array(out)

#     g = np.vectorize(lambda x: (x- min_x)/ (max_x - min_x))

X_train_hist_normed = normalize(X_train_histogram)
print('done1')
X_valid_hist_normed = normalize(X_valid_histogram)
print('done2')
X_test_hist_normed = normalize(X_test_histogram)
print('done3')

np.save('data/X_train_hist_normed.npy', X_train_hist_normed)
np.save('data/X_valid_hist_normed.npy', X_valid_hist_normed)
np.save('data/X_train_hist_normed.npy', X_test_hist_normed)
