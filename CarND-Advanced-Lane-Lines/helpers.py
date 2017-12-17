import numpy as np
import cv2
import glob

def img2Gray(filename):
    img = cv2.imread(filename)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
