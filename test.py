# reference
# https://pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/

import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils
import time

from utils.Toolbox import *
from utils.Camera.realsense import RealSense, D455_DEFAULT_COLOR, D455_DEFAULT_DEPTH, L515_DEFAULT_DEPTH, L515_DEFAULT_COLOR

# REALSENSE_SERIAL = "141322251060"
REALSENSE_SERIAL = "f1371347"

def nothing(x):
    pass

if __name__ == '__main__':

    cam = RealSense(serial=REALSENSE_SERIAL)
    cam.initialize(resolution_color=L515_DEFAULT_COLOR, resolution_depth=L515_DEFAULT_DEPTH)

    ballLower = (90, 95, 68)
    ballUpper = (127, 175, 194)

    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    radius_filtered = None
    while True:
        ts = time.time()

        img_rgb, map_depth, img_depth = cam.get_color_depth('cv')
        # img_rgb = cv2.resize(img_rgb, dsize=(320, 180), interpolation=cv2.INTER_AREA)

        # img_blur = cv2.GaussianBlur(img_rgb, (11, 11), 0)
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(img_hsv, ballLower, ballUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        center = None
        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # only proceed if the radius meets a minimum size
            if radius > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points

                if radius_filtered is None:
                    radius_filtered = radius

                radius_filtered = 0.7 * radius_filtered + 0.3 * radius

                cv2.circle(img_rgb, (int(x), int(y)), int(radius_filtered), (0, 255, 255), 2)
                cv2.circle(img_rgb, center, 5, (0, 0, 255), -1)
                cv2.putText(img_rgb, '{0:.3f}'.format(map_depth[int(y)//4, int(x)//4]), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (255, 0, 0), 4, cv2.LINE_AA)

                cv2.circle(img_depth, (int(x)//4, int(y)//4), int(radius_filtered)//4, (0, 255, 255), 1)
                cv2.circle(img_depth, center, 1, (0, 0, 255), -1)

        img_depth = cv2.resize(img_depth, dsize=D455_DEFAULT_DEPTH, interpolation=cv2.INTER_AREA)
        output = np.vstack((img_rgb, img_depth))
        output = cv2.resize(output, dsize=(640, 720), interpolation=cv2.INTER_AREA)

        tf = time.time()
        print(1/(tf-ts))

        cv2.imshow('RealSense', output)

        # Keyboard input
        k = cv2.waitKey(1) & 0xFF

        if k == 27:  # ESC
            cv2.destroyAllWindows()
            break