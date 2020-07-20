# import the necessary packages
import numpy as np
import argparse
import cv2
import imutils1

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image file")
ap.add_argument("-r", "--radius", type = int,
	help = "radius of Gaussian blur; must be odd")
args = vars(ap.parse_args())

# load the image and convert it to grayscale
image = cv2.imread(args["image"])
orig = image.copy()

resized = imutils1.resize(image, height=400)
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

# SUSCEPTIBLE METHOD
# the area of the image with the largest intensity value
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
cv2.circle(resized, maxLoc, 5, (255, 0, 0), 2)

# display the results of the naive attempt
cv2.imshow("Naive", resized)
cv2.waitKey(0)

cv2.imwrite("minmaxloc_susceptible.jpg", resized)
cv2.waitKey(0)

# ROBUST METHOD
# apply a Gaussian blur to the image then find the brightest
# region
gray = cv2.GaussianBlur(gray, (5, 5), 0)
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
cv2.circle(resized, maxLoc,5, (255, 0, 0), 2)

# display the results of our newly improved method
cv2.imshow("Robust",resized)
cv2.waitKey(0)

cv2.imwrite("minmaxloc_robust.jpg", resized)
cv2.waitKey(0)
