# computing packages
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import helpers
from moviepy.editor import VideoFileClip
# to take arguments from the input-- where to get the input video and where to save the output to
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input", help="input video file path",type=str)
parser.add_argument("--output", help="output video file path",type=str)
args = parser.parse_args()
input_video = args.input
output_video = args.output

# perspective transform matrices calculated previously
M = np.load('perspective_transform_matrix_2.npy')
Minv = np.load('perspective_transform_matrix_inv_2.npy')
# camera calibration calculated previous
camera_calibration = np.load('camera_calibration.npy').item()
dist = camera_calibration['dist']
mtx = camera_calibration['mtx']

# class for storing values about detected lane lines for video frames processing
class Line(object):
    def __init__(self, params):
        self.__dict__.update(params)
params = {
    # x values of the last n fits of the line
    'recent_xfitted' : [],
    #average x values of the fitted line over the last n iterations
    'bestx' : None,
    #polynomial coefficients averaged over the last n iterations
    'best_fit' : None ,
    #polynomial coefficients for the most recent fit
    'current_fit' : [np.array([False])],
    #radius of curvature of the line in some units
    'radius_of_curvature' : None,
    #distance in meters of vehicle center from the line
    'line_base_pos' : None,

}

# lane line processing pipeline
class detector:
    def __init__(self, camera_calibration, M, Minv, leftline, rightline,color_threshold_method='RGB',
                 lane_finding_method='histogram'):
        self.image= None
        self.input = None

        self.camera_calibration = camera_calibration
        self.color_threshold_method = color_threshold_method
        self.mtx = camera_calibration['mtx']
        self.dist = camera_calibration['dist']
        self.M = M
        self.Minv = Minv
        self.binary_image = None
        self.lane_finding_method = lane_finding_method
        self.leftline = leftline
        self.rightline = rightline
    # histogram equalisation for enhancing contrast
    def chlaheEqualisze(self, img):
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        imgs = []
        for i in range(img.shape[2]):
            img_chan = img[:,:,i]
            cl = clahe.apply(img_chan)
            imgs.append(cl)

        eq_img = np.stack(np.array(imgs), axis=0)
        eq_img = np.rollaxis(eq_img, 0, 3)
        return eq_img.copy()
    # sobel thresholding pipeline-- outputs the combination of the threshhold
    def sobel_thresholding(self, inputImg, sksize = 3,threshx = (15,90),threshy = (30, 90)
                               ,mag_thresh = (30,100), dir_thresh = (0.7, 1.3)):


        """
        # binary thresholding, best values found
        sksize = 3
        threshx = (15,90)
        threshy = (20, 90)
        mag_thresh = (30,100)
        dir_thresh = (0.7, 1.3)
        color_thresh = (70,255)
        """
        # x and y thresholding
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
        # magitude thresholding
        def mag_threshold(img, sobel_kernel=3, mag_thresh=(0, 255)):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
            mag = np.sqrt(np.square(sobelx) + np.square(sobely))
            scaled = np.uint8(255 * mag/np.max(mag))
            mag_binary = np.zeros_like(scaled)
            mag_binary[(scaled >= mag_thresh[0]) & (scaled <= mag_thresh[1])] = 1
            return mag_binary
        # direction thresholding
        def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
            abs_gradient = np.arctan2(np.absolute(sobely) ,np.absolute(sobelx))
            dir_binary = np.zeros_like(abs_gradient)
            dir_binary[(abs_gradient >= thresh[0]) & (abs_gradient <= thresh[1])] = 1
            return dir_binary
        # find the image binaries for each type of threshold
        def calculate_thresholded_imgs(inputImg, sksize = 3,threshx = (15,90),threshy = (30, 90)
                                   ,mag_thresh = (30,100), dir_thresh = (0.7, 1.3)):
            gradx = abs_sobel_thresh(inputImg, orient='x', sobel_kernel=sksize, thresh=threshx)
            grady = abs_sobel_thresh(inputImg, orient='y', sobel_kernel=sksize, thresh=threshy)
            mag_binary = mag_threshold(inputImg, sobel_kernel=sksize, mag_thresh=mag_thresh)
            dir_binary = dir_threshold(inputImg, sobel_kernel=sksize, thresh = dir_thresh)

            return gradx, grady, mag_binary, dir_binary
        # combine image binaries
        def calculate_combined_sobel(gradx, grady, mag_binary, dir_binary):
            combined = np.zeros_like(dir_binary)
            combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
            return combined

        gradx_img, grady_img, mag_binary_img, dir_binary_img = calculate_thresholded_imgs(inputImg, sksize ,threshx,threshy,mag_thresh, dir_thresh)
        return calculate_combined_sobel(gradx_img, grady_img, mag_binary_img, dir_binary_img)


    # combine both sobel and color thresholds
    def combined_thresholding(self, sobel_bin, color_binary):
        combined = np.zeros_like(sobel_bin)
        combined[(sobel_bin == 1) | (color_binary == 1)] = 1
        return combined

    # color thresolding
    def color_thresholding_pipeline(self, img):
        def color_threshold(img, thresh=(0,255), cspace= 'HLS'):
            if cspace== 'HLS':
                img_transformed = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
            else:
                img_transformed = img
            channel = img_transformed[:,:,2] #if using the R channel it's also the last one
            binary_output = np.zeros_like(channel)
            binary_output[(channel >= thresh[0]) & (channel <= thresh[1])] = 1
            return binary_output
        def calculate_combined_color(hls_binary, bgr_binary):
            combined = np.zeros_like(hls_binary)
            combined[(hls_binary == 1) & (bgr_binary == 1)] = 1
            return combined

        hls_binary = color_threshold(img, thresh=(70,255), cspace= 'HLS')
        bgr_binary = color_threshold(img, thresh=(150,255), cspace= 'RGB')

        return  calculate_combined_color(hls_binary, bgr_binary), hls_binary, bgr_binary
    # histogram detection for lane lines
    def histogram_method(self, binary_image):
        histogram = np.sum(binary_image[binary_image.shape[0]//2:,:], axis=0)
        midpoint = np.int(histogram.shape[0]/2)
        left_peak = np.argmax(histogram[:midpoint])
        right_peak = np.argmax(histogram[midpoint:]) + midpoint

        nwindows = 9
        window_height = np.int(binary_image.shape[0] / nwindows)

        # find where all the potenial lane pixels are
        nonzero = binary_image.nonzero()
        nonzero_x = np.array(nonzero[1])
        nonzero_y = np.array(nonzero[0])

        # make copy of image to work with
        im= np.copy(binary_image)
        img_cur = np.stack((im, im, im), -1) * 255
        # set the peak positions
        leftpeak_curr = left_peak
        rightpeak_curr = right_peak

        margin= 150
        minpix =30

        leftlaneidx = []
        rightlaneidx = []

        for i in range(nwindows):
            #get coordinates of the windows
            win_y_bottom = img_cur.shape[0] - (i + 1) * window_height
            win_y_top = img_cur.shape[0] - i * window_height
            win_x_left_low = leftpeak_curr - margin
            win_x_left_high = leftpeak_curr + margin
            win_x_right_low = rightpeak_curr - margin
            win_x_right_high = rightpeak_curr + margin

            left_indexes = ((nonzero_y >= win_y_bottom) & (nonzero_y < win_y_top) &
                            (nonzero_x >= win_x_left_low) & (nonzero_x < win_x_left_high)).nonzero()[0]
            right_indexes = ((nonzero_y >= win_y_bottom) & (nonzero_y < win_y_top) &
                            (nonzero_x > win_x_right_low) & (nonzero_x < win_x_right_high)).nonzero()[0]

            leftlaneidx.append(left_indexes)
            rightlaneidx.append(right_indexes)

            if len(left_indexes) > minpix:
                leftpeak_curr = np.int(np.mean(nonzero_x[left_indexes]))
                cv2.rectangle(img_cur, (win_x_left_low, win_y_bottom), (win_x_left_high, win_y_top), (0,255,0), 2)
            if len(right_indexes) > minpix:
                rightpeak_curr = np.int(np.mean(nonzero_x[right_indexes]))
                cv2.rectangle(img_cur, (win_x_right_low, win_y_bottom), (win_x_right_high, win_y_top), (0,255,0), 2)

        leftlaneidx = np.concatenate(leftlaneidx)
        rightlaneidx = np.concatenate(rightlaneidx)

        leftx = nonzero_x[leftlaneidx]
        lefty = nonzero_y[leftlaneidx]
        rightx = nonzero_x[rightlaneidx]
        righty = nonzero_y[rightlaneidx]

        return leftx,lefty, rightx, righty
    # calculate the radius of curvature
    def radius_of_curvature_calc(self, left_xvals,right_xvals, y_vals, y_eval ):
        ym_per_pix = 30/360 # meters per pixel in y dimension
        xm_per_pix = 3.7/950 # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(y_vals * ym_per_pix, left_xvals*xm_per_pix, 2)
        right_fit_cr = np.polyfit(y_vals * ym_per_pix, right_xvals*xm_per_pix, 2)
        # radius of curvature in real space
        left_curverad_actual = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad_actual = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        left_xval = left_fit_cr[2] + left_fit_cr[1] *y_eval*ym_per_pix + left_fit_cr[0] *(y_eval*ym_per_pix )**2
        right_xval = right_fit_cr[2] + right_fit_cr[1] *y_eval*ym_per_pix + right_fit_cr[0] *(y_eval*ym_per_pix ) **2

        offset_from_center = 0.5 * (left_xval + right_xval) - 0.5 * xm_per_pix * self.image.shape[1]
        return left_curverad_actual, right_curverad_actual, offset_from_center

    # sanity checking the found values-- if sanity passes use the found values for that frame in further calcs, otherwise reject
    def sanity_check(self, left_polyfit, right_polyfit, left_curverad, right_curverad, y_eval):
        ym_per_pix = 3.7/950
        lanewidth_top = (right_polyfit[2] - left_polyfit[2]) * ym_per_pix
        lanewidth_bottom =((right_polyfit[2] + right_polyfit[1] *y_eval + right_polyfit[0] *y_eval**2)
                           - (left_polyfit[2] + left_polyfit[1] *y_eval + left_polyfit[0] *y_eval**2)) * ym_per_pix

        if lanewidth_top < 2 or lanewidth_top > 5:
            return False
        if lanewidth_bottom < 2 or lanewidth_bottom > 5:
            return False
        if left_curverad / right_curverad > 2 or left_curverad / right_curverad < 0.5:
            return False

        return True

    def pipeline(self, imgInput):
        self.image = imgInput
        undistorted = cv2.undistort(imgInput, self.mtx, self.dist, None, self.mtx)
        imgInput_cv2 = cv2.cvtColor(undistorted, cv2.COLOR_RGB2BGR)
        img = cv2.GaussianBlur(undistorted,(3,3),0)
        img = self.chlaheEqualisze(img)
        color_bin_combined, hls_binary, bgr_binary = self.color_thresholding_pipeline(img)
        if self.color_threshold_method =='RGB':
            color_bin = bgr_binary
        elif self.color_threshold_method == 'HLS':
            color_bin = hls_binary
        else :
            color_bin = color_bin_combined

        sobel_bin = self.sobel_thresholding(imgInput)
        binary_img = self.combined_thresholding(sobel_bin, color_bin)

        #project to birds eye view:
        topview = cv2.warpPerspective(binary_img, M, (binary_img.shape[1], binary_img.shape[0]), flags=cv2.INTER_LINEAR)
        topview_img = np.stack([topview, topview, topview], axis = -1) #* 255

#         if self.lane_finding_method== 'histogram':
        leftx,lefty, rightx, righty = self.histogram_method(topview)

       # todo inplemenrt sliding widnow method


        # Fit a second order polynomial to each
        left_polyfit = np.polyfit(lefty, leftx, 2)
        right_polyfit = np.polyfit(righty, rightx, 2)

        plotvals = np.linspace(0, topview.shape[0] - 1, topview.shape[0])
        left_xvals = left_polyfit[2] + left_polyfit[1] * plotvals + left_polyfit[0] * plotvals **2
        right_xvals = right_polyfit[2] + right_polyfit[1] * plotvals + right_polyfit[0] * plotvals **2

        left_curverad_actual_curr, right_curverad_actual_curr, offset_curr = self.radius_of_curvature_calc(left_xvals,right_xvals, plotvals, topview.shape[0] )

        is_detection_okay = self.sanity_check(left_polyfit, right_polyfit, left_curverad_actual_curr, right_curverad_actual_curr, self.image.shape[0])

        if is_detection_okay == True or len(self.leftline.recent_xfitted) == 0:
            self.leftline.recent_xfitted.append(left_xvals)
            self.rightline.recent_xfitted.append(right_xvals)

        if len(self.leftline.recent_xfitted) > 5:
            self.leftline.recent_xfitted = self.leftline.recent_xfitted[-5:]
        if len(self.rightline.recent_xfitted) > 5:
            self.rightline.recent_xfitted = self.rightline.recent_xfitted[-5:]

        self.leftline.bestx = np.average( np.array(self.leftline.recent_xfitted), axis=0)
        self.rightline.bestx = np.average( np.array(self.rightline.recent_xfitted), axis=0)

        left_curverad_actual, right_curverad_actual, offset = self.radius_of_curvature_calc(self.leftline.bestx,self.rightline.bestx, plotvals, topview.shape[0] )
        img_with_lanes = helpers.draw_lane_lines(undistorted, self.leftline.bestx, self.rightline.bestx, plotvals, self.Minv)


        #  print the radius of curvature etc onto images
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_with_lanes,'Radius of curvature:',(10,50), font, 1,(0,255,0),2,cv2.LINE_AA)
        cv2.putText(img_with_lanes,'Left: ' + str(round(left_curverad_actual/1000, 2)) + ' km',(10,100), font, 1,(0,255,0),2,cv2.LINE_AA)
        cv2.putText(img_with_lanes,'Right: ' +str(round(right_curverad_actual/1000, 2)) + ' km', (10,150), font, 1,(0,255,0),2,cv2.LINE_AA)
        cv2.putText(img_with_lanes,'offset: ' +str(round(offset, 2)) + ' m', (10,200), font, 1,(0,255,0),2,cv2.LINE_AA)

        return cv2.cvtColor(img_with_lanes, cv2.COLOR_BGR2RGB)


leftline = Line(params)
rightline = Line(params)
lane_detector = detector(camera_calibration, M, Minv,leftline, rightline, color_threshold_method='combined',
                 lane_finding_method='histogram')
clip1 = VideoFileClip(input_video)
clipout = clip1.fl_image(lane_detector.pipeline)
clipout.write_videofile(output_video, audio=False, verbose=False, progress_bar=False)
