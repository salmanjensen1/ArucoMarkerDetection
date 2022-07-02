from imutils.video import VideoStream
import imutils
import cv2
import sys
from Aruco_Dict import ARUCO_DICT
import math
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
        vs = VideoStream(src=0).start()
        # capture = cv2.VideoCapture(0)
        # time.sleep(2.0)

        # loop over the frames from the video stream
        while True:
            # grab the frame from the threaded video stream and resize it
            # to have a maximum width of 1000 pixels
            frame = vs.read()
            frame = imutils.resize(frame, width=1000)
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
                    distance = (real_face_width * focal_length) / width_in_rf_image
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
        vs.stop()