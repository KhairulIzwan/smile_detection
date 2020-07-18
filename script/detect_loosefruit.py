#!/usr/bin/env python3

#USAGE
#Python detect_loosefruit.py combination of two methods
#1. color analysis
#2. CNN deep learning --model output/lenet.hdf5

#Import the necessary packages

#ImportError : no module named 'cv2' python3

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
from skimage import measure
from imutils import contours

import os

count = 0

#construct the argument parse and parse the argument

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to pre-trained smile detector CNN")
ap.add_argument("-i", "--image", required=True, help="path to the image")
args = vars(ap.parse_args())

#load the color_analysis

# STEP 1: Load an image
image = cv2.imread(args["image"])

#STEP 1a: copy original
image_orig = image.copy()

#STEP 2 : colorspace(convert image to hsv)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#STEP 3: Masking
#range for lower red
lower_red = np.array([0,120,70])
upper_red = np.array([10,255,255])
mask1 = cv2.inRange(hsv, lower_red, upper_red)

#range for upper red
lower_red = np.array([170,120,70])
upper_red = np.array([180,255,255])
mask2 = cv2.inRange(hsv, lower_red, upper_red)

#generating the final mask to detect the red color
mask= mask1 + mask2
cv2.imshow("final mask", mask)

#apply the filtering process
mask = cv2.erode(mask, None, iterations=1)
mask = cv2.dilate(mask, None, iterations=1)
cv2.imshow("after filtering", mask)

#segmentating the cloth out of the frame using bitwise and woth the invterted mask
res1 = cv2.bitwise_and(image, mask=mask)
cv2.imshow("res", res1)

#STEP 4 : counting and labelling
labels = measure.label(mask, neighbors=8, background=0)
mask = np.zeros(mask.shape, dtype="uint8")

#LOAD MODEL=LENET.HDF5
model = load_model(args["model"])

while True :

    # STEP 1: Load an image
    image = cv2.imread(args["image"])

    #STEP 1a: copy original
    image_orig = image.copy()

    #STEP 2 : colorspace(convert image to hsv)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0,120,70])
    upper_red = np.array([10,255,255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    #range for upper red
    lower_red = np.array([170,120,70])
    upper_red = np.array([180,255,255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    #generating the final mask to detect the red color
    mask= mask1 + mask2
    cv2.imshow("final mask", mask)

    #apply the filtering process
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)
    cv2.imshow("after filtering", mask)

    labels = measure.label(mask, neighbors=8, background=0)
    mask = np.zeros(mask.shape, dtype="uint8")

    for label in np.unique(labels):
	# if this is the background label, ignore it
	   if label == 0:
		         continue
	   # otherwise, construct the label mask and count the
	   # number of pixels
	   labelMask = np.zeros(mask.shape, dtype="uint8")
	   labelMask[labels == label] = 255
	   numPixels = cv2.countNonZero(labelMask)
	# if the number of pixels in the component is sufficiently

	# large, then add it to our mask of "large blobs"
	   if numPixels > 185:
	          mask = cv2.add(mask, labelMask)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
	   cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = contours.sort_contours(cnts)[0]

    for (i, c) in enumerate(cnts):
         # draw the bright spot on the image
         (x, y, w, h) = cv2.boundingRect(c)
         ((cX, cY), radius) = cv2.minEnclosingCircle(c)

         print(x, y, w, h, image.shape[1], image.shape[0])

         roi = image_orig[y:y + h, x:x + w]
         roi = cv2.resize(roi,(28,28))
         roi = img_to_array(roi)
         roi = np.expand_dims(roi, axis=0)

         (notFruit, fruit) = model.predict(roi)[0]
         label = "Fruit" if fruit > notFruit else "Not fruit"

          cv2.circle(image, (int(cX), int(cY)), int(radius), (255, 255, 255), 2)
          cv2.putText(image, "#{}".format(i + 1), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

         cv2.imshow("Fruit", image_orig)

    if cv2.waitKey(1) & 0xFF == ord("q"):
    	break

cv2.destroyAllWindows()
