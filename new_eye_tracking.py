"""================================================================================
  *
  * Author     : Duy Le, Arya Nguyen
  * Description: Track eyes and head pose
  *
================================================================================"""

import cv2
import dlib
import numpy as np
import pyautogui as pg
import os
import sys
import time
from pathlib import Path

from utils import *

# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')


# Define properties
blinking_ratio = 0
gaze_ratio = 0
last_time = 0
filtered_X = 0
filtered_Y = 0
T = time.time()


def main():
    device = dlib.cuda.get_num_devices()

    # Load: file that contains trained facial shape facial_predictor
    trained_facial_shape_predictor = os.path.join(
        Path(__file__).parent.absolute(),
        'trained_facial_model/shape_predictor_68_face_landmarks.dat'
    )

    # Capture video from camera
    cap = cv2.VideoCapture(0)

    # Use for detect face, return the rectangle the envelop the face
    detector = dlib.get_frontal_face_detector()

    # Predict 68 points on face
    facial_predictor = dlib.shape_predictor(
        trained_facial_shape_predictor
    )

    x_size, y_size = pg.size()  # current screen resolution width and height
    pg.moveTo(int(x_size) / 2, int(y_size) / 2)  # move to the middle of the screen
    pg.FAILSAFE = False

    while True:
        _, frame = cap.read()  # Grab the frame from the threaded video file stream
        # frame = cv2.imread("headPose.jpg")
        frame = cv2.flip(frame, 1)  # flip frame vertically
        new_frame = np.zeros((500, 500, 3), np.uint8)  # resize frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert frame to grayscale

        # get all bounding boxes around all the faces that can be found in the frame
        all_faces = detector(gray_frame)

        # handling multiple faces: get the one that is the closest to the center of the frame
        if len(all_faces) == 1:
            face = all_faces[0]
        else:
            pass

        # x, y = face.left(), face.top()
        # x1, y1 = face.right(), face.bottom()
        # cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        # get facial landmarks
        facial_landmarks = facial_predictor(gray_frame, face)

        # Get Image points
        image_points = get_image_points(
            [30, 8, 36, 45, 48, 54],
            facial_landmarks
        )
        endpoint, origin = headpose(frame, image_points)
        X, Y = control_mouse(endpoint, frame)

        # Detect blinking
        left_eye_ratio = get_blinking_ratio(
            [36, 37, 38, 39, 40, 41],
            facial_landmarks
        )
        # right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], facial_landmarks)
        # blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
        blinking_ratio = left_eye_ratio

        # Gaze detection of one
        gaze_ratio_left_eye = get_gaze_ratio(
            frame,
            gray_frame,
            [36, 37, 38, 39, 40, 41],
            facial_landmarks
        )
        # gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], facial_landmarks)
        # gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2
        gaze_ratio = gaze_ratio_left_eye

        print(blinking_ratio)
        #
        if blinking_ratio > 6 and time.time() - last_time > 1:
            print_text_to_frame(frame, "CLICK", font_scale=7, color=(255, 0, 0), thickness=5)
            pg.click()
            time.sleep(1)
            last_time = time.time()
        # Gaze to the right
        if gaze_ratio <= 0.8:
            print_text_to_frame(frame, "RIGHT")
            new_frame[:] = (0, 0, 255)
        # Pupil in the center
        elif 1 < gaze_ratio < 2.0:
            print_text_to_frame(frame, "CENTER")
        # Gaze to the left
        else:
            new_frame[:] = (255, 0, 0)
            print_text_to_frame(frame, "LEFT")

        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("Frame", frame)  # display frame

        key = cv2.waitKey(1)
        if key == 27:
            break

        cap.release()  # close camera
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
