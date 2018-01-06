import numpy as np
import cv2
import glob

def img2Gray(filename):
    img = cv2.imread(filename)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def chlaheEqualisze(img):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    imgs = []
    for i in range(img.shape[2]):
        img_chan = img[:,:,i]
        cl = clahe.apply(img_chan)
        imgs.append(cl)

    eq_img = np.stack(np.array(imgs), axis=0)
    eq_img = np.rollaxis(eq_img, 0, 3)
    output = eq_img.copy()
    return output

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

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
        """
        `img` is an image with lines drawn on it.
        Should be a blank image (all black) with lines drawn on it.

        `initial_img` should be the image before any processing.

        The result image is computed as follows:

        initial_img * α + img * β + λ
        NOTE: initial_img and img must be the same shape!
        """
        return cv2.addWeighted(initial_img, α, img, β, λ)

def draw_lane_lines(imgInput, left_xvals, right_xvals, yvals, Minv):
    line_canvas = np.zeros_like(imgInput)
    line_coords_left  = np.int32(np.stack((left_xvals, yvals), axis = -1))
    line_coords_right  = np.int32(np.stack((right_xvals, yvals), axis = -1))

    line_canvas = cv2.polylines(line_canvas, [line_coords_left], False, (255, 0,0), 20)
    line_canvas = cv2.polylines(line_canvas, [line_coords_right], False, (255, 0,0), 20)
    points = np.concatenate((line_coords_left, np.flipud(line_coords_right)))
    cv2.fillPoly(line_canvas, [points], color=[0,255,0])
    line_canvas_inv = cv2.warpPerspective(line_canvas, Minv, (line_canvas.shape[1], line_canvas.shape[0]), flags=cv2.INTER_LINEAR)
    img_with_lanes = weighted_img(line_canvas_inv, imgInput, 1., 0.4)

    return cv2.cvtColor(img_with_lanes, cv2.COLOR_BGR2RGB)
