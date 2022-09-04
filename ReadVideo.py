from collections import deque

import numpy as np
import pyrealsense2
from imutils.video import VideoStream
import imutils
import cv2
import sys
from Aruco_Dict import ARUCO_DICT
import math
from realsense_depth import *

class ReadVideo(object):

    def calculateDistance(self, x1, y1, x2, y2):
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return dist

    def readVideo(self, type):
        if ARUCO_DICT.get(type, None) is None:
            print("[INFO] ArUCo tag of '{}' is not supported".format(
                type))
            sys.exit(0)
        # load the ArUCo dictionary and grab the ArUCo parameters
        print("[INFO] detecting '{}' tags...".format(type))
        arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[type])
        arucoParams = cv2.aruco.DetectorParameters_create()
        # initialize the video stream and allow the camera sensor to warm up
        print("[INFO] starting video stream...")
        #******************************READ VIDEO STREAM*************************************************
        # vs = VideoStream(src=1).start()
        dc = DepthCamera()
        ret, frame, color_frame = dc.get_frame()

        # capture = cv2.VideoCapture(0)
        # time.sleep(2.0)
        #*****************************INITIALIZATION FOR TRACKED POINTS************************************************************
        # initialize the list of tracked points, the frame counter,
        # and the coordinate deltas
        pts = deque(maxlen=32)
        counter = 0
        (dX, dY) = (0, 0)
        #**************************************************************************************************************************
        # loop over the frames from the video stream
        while True:
            # grab the frame from the threaded video stream and resize it
            # to have a maximum width of 1000 pixels
            # frame = vs.read()
            frame = imutils.resize(frame, width=1000)
            (h, w) = frame.shape[:2]
            (sX, sY) = (w // 2, h // 2)
            cv2.circle(frame, (sX, sY), 4, (0, 255, 0), -1)
            # detect ArUco markers in the input frame
            (corners, ids, rejected) = cv2.aruco.detectMarkers(frame,
                                                               arucoDict, parameters=arucoParams)
            if len(corners) > 0:
                # flatten the ArUco IDs list
                ids = ids.flatten()
                # loop over the detected ArUCo corners
                for (markerCorner, markerID) in zip(corners, ids):
                    # extract the marker corners (which are always returned
                    # in top-left, top-right, bottom-right, and bottom-left
                    # order)
                    corners = markerCorner.reshape((4, 2))
                    (topLeft, topRight, bottomRight, bottomLeft) = corners
                    # convert each of the (x, y)-coordinate pairs to integers
                    topRight = (int(topRight[0]), int(topRight[1]))
                    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                    topLeft = (int(topLeft[0]), int(topLeft[1]))

                    width_in_rf_image = math.sqrt((bottomRight[0] - bottomLeft[0]) ** 2 + (bottomRight[1] - bottomRight[1]) ** 2)
                    real_width = 100
                    measured_distance = 40
                    real_face_width = 100
                    # print(width_in_rf_image)
                    # focal_length = (width_in_rf_image * measured_distance) / real_width
                    focal_length = 90.8
                    # print("*******")
                    # print(focal_length)
                    # print("*******")
                    if(width_in_rf_image!=0):
                        distance = (real_face_width * focal_length) / width_in_rf_image
                    center_distance = 40
                    perpendicular_distance = pow(distance,2) - pow(center_distance, 2)
                    if(perpendicular_distance>=0):
                        distance_moved = math.sqrt(perpendicular_distance)
                    # print("{:.2f}".format(distance_moved) + " units") #*********************************************************
                    # print(distance)
                    # draw the bounding box of the ArUCo detection
                    cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
                    cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
                    cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
                    cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)
                    # compute and draw the center (x, y)-coordinates of the
                    # ArUco marker
                    cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                    cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                    distance = frame[cX, cY]
                    center = (cX, cY)
                    pts.appendleft(center)
                    print(len(pts))
                    for i in np.arange(1, len(pts)):
                        # if either of the tracked points are None, ignore
                        # them
                        if pts[i - 1] is None or pts[i] is None:
                            continue
                        # check to see if enough points have been accumulated in
                        # the buffer
                        if counter >= 10 and i == 1 and len(pts) == 32:
                            # compute the difference between the x and y
                            # coordinates and re-initialize the direction
                            # text variables
                            dX = pts[i-10][0] - pts[i][0]
                            dY = pts[i-10][1] - pts[i][1]
                            (dirX, dirY) = ("", "")
                            # ensure there is significant movement in the
                            # x-direction
                            if np.abs(dX) > 20:
                                dirX = "East" if np.sign(dX) == 1 else "West"
                            # ensure there is significant movement in the
                            # y-direction
                            if np.abs(dY) > 20:
                                dirY = "North" if np.sign(dY) == 1 else "South"
                            # handle when both directions are non-empty
                            if dirX != "" and dirY != "":
                                direction = "{}-{}".format(dirY, dirX)
                            # otherwise, only one direction is non-empty
                            else:
                                direction = dirX if dirX != "" else dirY
                        # cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        #             0.65, (0, 0, 255), 3)
                        cv2.putText(frame, "dx: {}, dy: {}".format(dX, dY),
                                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.35, (0, 0, 255), 1)
                        counter+=1
                    cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
                    # draw the ArUco marker ID on the frame
                    cv2.putText(frame, str(markerID),
                                (topLeft[0], topLeft[1] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)
                    cv2.putText(frame, str(distance),
                                (bottomRight[0], bottomLeft[1] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)
            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(20) & 0xFF
             # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
        # do a bit of cleanup
        cv2.destroyAllWindows()
        frame.stop()