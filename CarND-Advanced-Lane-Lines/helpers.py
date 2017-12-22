import numpy as np
import cv2
import glob

def img2Gray(filename):
    img = cv2.imread(filename)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        x1 = line[0]
        y1 = line[1]
        x2 = line[2]
        y2 = line[3]
#         for x1,y1,x2,y2 in line:
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    return img

def order_points(points):
    """
    ensure that the pts list are ordered from top left in clockwise
    direction for perspectiveTransform
    """
    rect = np.zeros((4, 2), dtype = "float32")
    coordsSum = np.sum(points, axis=1)
    coordsDiff = np.diff(points, axis=1)
    # top left point has smallest sum
    rect[0] = points[np.argmin(coordsSum)]
    # bottom right point has biggest sum
    rect[2] = points[np.argmax(coordsSum)]
    # top right point has smallest (in fact, can be -ve difference)
    rect[1] = points[np.argmin(coordsDiff)]
    #  and the bottom left the highest
    rect[3] = points[np.argmax(coordsDiff)]
	# return the ordered coordinates

    return rect


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobel = np.absolute(sobel)
    scaled = np.uint8(255 * abs_sobel/np.max(abs_sobel))
    grad_binary = np.zeros_like(scaled)
    grad_binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
    return grad_binary

def mag_threshold(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    mag = np.sqrt(np.square(sobelx) + np.square(sobely))
    scaled = np.uint8(255 * mag/np.max(mag))
    mag_binary = np.zeros_like(scaled)
    mag_binary[(scaled >= mag_thresh[0]) & (scaled <= mag_thresh[1])] = 1
    return mag_binary

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_gradient = np.arctan2(np.absolute(sobely) ,np.absolute(sobelx))
    dir_binary = np.zeros_like(abs_gradient)
    dir_binary[(abs_gradient >= thresh[0]) & (abs_gradient <= thresh[1])] = 1
    return dir_binary
