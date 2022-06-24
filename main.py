import cv2
import numpy
import argparse
from ReadImage import ReadImage
from ReadVideo import ReadVideo
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to image")
ap.add_argument("-t", "--type", type=str,
                default="DICT__ORIGINAL",  # cv2.aruco Dict holds a list of different cv2.aruco markers
                help="type of cv2.aruco tag to detect")  # type of the cv2.aruco marker
args = vars(ap.parse_args())
imageReadObject = ReadImage();
# imageReadObject.readImage(args["image"], args["type"])

videoReadObject = ReadVideo();
videoReadObject.readVideo(args["type"])
