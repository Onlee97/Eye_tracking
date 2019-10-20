"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2
import numpy as np
from gaze_tracking import GazeTracking


def lowpass_filter(x, old_x, alpha):
    filterd_x = alpha*x + (1-alpha)*old_x
    return filterd_x

def clamp(x, _min, _max):
    if x < _min:
        x = 0
    elif x > _max:
        x = 1
    return x


def main():
    gaze = GazeTracking()
    webcam = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    horizontal_ratio = 0
    vertical_ratio = 0

    hr_filtered = 0
    vr_filtered = 0

    cursor_x = 0
    cursor_y = 0

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
        # h = h*2-1
        # v = v*2-1


        # horizontal_ratio = clamp(h,-1,1)
        # vertical_ratio = clamp(v,-1,1)

        if h != -1:
            horizontal_ratio = clamp(h,0,1)
            vertical_ratio = clamp(v,0,1)

        alpha = 0.1
        hr_filtered = lowpass_filter(horizontal_ratio, hr_filtered, alpha)
        vr_filtered = lowpass_filter(vertical_ratio, vr_filtered, alpha)

        print(hr_filtered, vr_filtered)
        

        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()
        cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)


        width = 1000
        height = 500
        # blank_image = np.zeros((height, width))
        blank_image = np.zeros(shape=[height, width, 3], dtype=np.uint8)
        cv2.imshow("blank", blank_image)

        cursor_x = int(width*vr_filtered)
        cursor_y = int(height*hr_filtered)

        cursor_x = clamp(cursor_x, 0, width)
        cursor_y = clamp(cursor_y, 0, height)


        # cursor_x = int(width*0.5)
        # cursor_y = int(height*0.5)

        radius = 10
        cv2.circle(blank_image, (cursor_x, cursor_y), radius, [255,255,255], 2)

        cv2.imshow("blank", blank_image)

        frame = cv2.flip(frame, 1)
        cv2.imshow("Demo", frame)

        if cv2.waitKey(1) == 27:
            break

if __name__ == "__main__":
    main()