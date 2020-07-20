import numpy as np
import argparse
import imutils
import cv2
import os

count = 1

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
ap.add_argument("-r", "--raw", required=True, help="Path to save the image")
args = vars(ap.parse_args())


image = cv2.imread(args["image"])
cv2.imshow("Original", image)

(h, w) = image.shape[:2]
center = (w // 2, h // 2)

M = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("Rotated by 45 Degrees", rotated)

dirPath = os.path.sep.join([args["raw"]])
if not os.path.exists(dirPath):
    os.makedirs(dirPath)

p = os.path.sep.join([dirPath, "{}.png".format(str(count).zfill(6))])
cv2.imwrite(p, rotated)


# M = cv2.getRotationMatrix2D(center, -90, 1.0)
# rotated = cv2.warpAffine(image, M, (w, h))
# cv2.imshow("Rotated by -90 Degrees", rotated)
