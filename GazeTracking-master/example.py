"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2
import numpy as np
from gaze_tracking import GazeTracking

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cursor_horizontal_ratio = 0
cursor_vertical_ratio = 0
while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()
    # face_frame = detect_faces(frame, face_cascade)
    # if face_frame is not None:
    #     eyes = detect_eyes(face_frame, eye_cascade)
    # # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    text = ""

    # if gaze.is_blinking():
    #     text = "Blinking"
    # elif gaze.is_right():
    #     text = "Looking right"
    # elif gaze.is_left():
    #     text = "Looking left"
    # elif gaze.is_center():
    #     text = "Looking center"

    # cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
    h, v = gaze.position()
    if h != -1:
        cursor_horizontal_ratio = h
        cursor_vertical_ratio = v

    print(cursor_horizontal_ratio, cursor_vertical_ratio)
    

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)


    # width = 500
    # height = 250
    # # blank_image = np.zeros((height, width))
    # blank_image = np.zeros(shape=[height, width, 3], dtype=np.uint8)
    # cv2.imshow("blank", blank_image)

    # # cursor_x = int(width*cursor_vertical_ratio)
    # # cursor_y = int(height*cursor_horizontal_ratio)

    # cursor_x = int(width*0.5)
    # cursor_y = int(height*0.5)

    # radius = 20
    # cv2.circle(blank_image, (cursor_x, cursor_y), radius, [0,0,0], 2)
    # cv2.imshow("blank", blank_image)
    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break
