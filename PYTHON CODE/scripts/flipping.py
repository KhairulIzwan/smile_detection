from imutils import paths
import argparse
import imutils
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,help="path to input directory of images")
ap.add_argument("-a", "--annot", required=True,help="path to output directory of annotations")
ap.add_argument("-f", "--folder", required=True, help="folder name")
# ap.add_argument("-i", "--image", required = True, help = "Path to the image")
# ap.add_argument("-r", "--raw", required=True, help="Path to save the image")
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args["input"]))
count = 0

for (i, imagePath) in enumerate(imagePaths):
    print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
    image = cv2.imread(imagePaths)

    flipped = cv2.flip(image, 0)
# cv2.imshow("Flipped Horizontally", flipped)

# flipped = cv2.flip(image, 0)
# cv2.imshow("Flipped Vertically", flipped)
#
# flipped = cv2.flip(image, -1)
# cv2.imshow("Flipped Horizontally & Vertically", flipped)

    dirPath = os.path.sep.join([args["raw"]])
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

    p = os.path.sep.join([dirPath, "{}.png".format(str(count).zfill())])
    cv2.imwrite(p, flipped)

    count = count + 1

# except KeyboardInterrupt:
# 	print("[INFO] manually leaving script")
# 	break
#
# except:
# 	print("[INFO] skipping image...")
