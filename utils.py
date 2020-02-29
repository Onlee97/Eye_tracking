"""================================================================================
  *
  * Author     : Duy Le, Arya Nguyen
  * Description: Utilities function
  *
================================================================================"""
import cv2
import numpy as np
import pyautogui as pg
from math import hypot


def midpoint(p1, p2):
    """

    :param p1:
    :param p2:
    :return:
    """
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


def get_blinking_ratio(eye_points, facial_landmarks):
    """ Get blinking ratio from the eyes

    :param eye_points:
    :param facial_landmarks: Facial landmarks

    :return:
    """
    #
    left_point = (
        facial_landmarks.part(eye_points[0]).x,  # x-coordinate of the eye point
        facial_landmarks.part(eye_points[0]).y  # y-coordinate of the eye point
    )
    right_point = (
        facial_landmarks.part(eye_points[3]).x,
        facial_landmarks.part(eye_points[3]).y
    )
    center_top = midpoint(
        facial_landmarks.part(eye_points[1]),
        facial_landmarks.part(eye_points[2])
    )
    center_bottom = midpoint(
        facial_landmarks.part(eye_points[5]),
        facial_landmarks.part(eye_points[4])
    )

    # hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    # ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    # math.hypot return length of the vector from the origin to point (x, y) = sqrt(x*x + y*y)
    hor_line_length = hypot(
        (left_point[0] - right_point[0]),
        (left_point[1] - right_point[1])
    )
    ver_line_length = hypot(
        (center_top[0] - center_bottom[0]),
        (center_top[1] - center_bottom[1])
    )

    ratio = hor_line_length / ver_line_length
    return ratio


def get_gaze_ratio(frame, gray_frame, eye_points, facial_landmarks):
    """ Get gaze ratio of the eyes

    :param frame:
    :param gray_frame:
    :param eye_points:
    :param facial_landmarks:

    :return: gaze ratio
    """
    left_eye_region = np.array(
        [(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
         (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
         (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
         (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
         (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
         (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)
         ],
        np.int32)

    # Draw the eyes
    cv2.polylines(
        frame,
        [left_eye_region],
        True,
        (255, 255, 255),
        2
    )

    height, width, _ = frame.shape
    mask = np.zeros(
        (height, width),
        np.uint8
    )
    cv2.polylines(
        mask,
        [left_eye_region],
        True,
        255,
        2
    )
    cv2.fillPoly(
        mask,
        [left_eye_region],
        255
    )
    eye = cv2.bitwise_and(
        gray_frame,
        gray_frame,
        mask=mask
    )

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(
        gray_eye,
        70,
        255,
        cv2.THRESH_BINARY
    )
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(
        left_side_threshold
    )
    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white

    return gaze_ratio


def clamp(x, _min, _max):
    """ Keep mouse cursor in the screen

    :param x: cursor current position
    :param _min: lowest point of the screen
    :param _max: highest point of the screen

    :return: x
    """
    if x < _min:
        x = _min
    elif x > _max:
        x = _max
    return x


def get_image_points(int_array, facial_landmarks):
    """

    :param int_array:
    :param facial_landmarks: facial landmarks
    :return:
    """
    image_points = []
    for i in int_array:
        image_points.append(
            (facial_landmarks.part(i).x,
             facial_landmarks.part(i).y)
        )
    return image_points


def headpose(img, image_points):
    """

    :param img: an image/ a frame
    :param image_points:

    :return:
    """
    # Nose tip
    # Chin
    # Left eye left corner
    # Right eye right corne
    # Left Mouth corner
    # Right mouth corner
    size = img.shape

    # 2D image points. If you change the image, you need to change vector
    image_points = np.array(image_points, dtype="double")
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ])

    # Camera internals
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ],
        dtype="double"
    )

    # print("Camera Matrix : ",format(camera_matrix))

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=0
    )

    # print("Rotation Vector: ",format(rotation_vector))
    # print("Translation Vector: ",format(translation_vector))

    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose
    Start_point = [0.0, 170, -135]  # The point in the middle of the eye

    (nose_end_point2D, jacobian) = cv2.projectPoints(
        np.array(
            [(0.0, 0.0, 1000.0)]
        ),
        rotation_vector,
        translation_vector,
        camera_matrix,
        dist_coeffs)
    # print(nose_end_point2D)
    for p in image_points:
        cv2.circle(
            img,
            (int(p[0]), int(p[1])),
            3,
            (255, 255, 255),
            -1
        )

    # p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    # p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    # cv2.line(img, p1, p2, (255,255,255), 2)

    return nose_end_point2D[0][0], image_points[0]


def lowpass_filter(x, old_x, alpha):
    """

    :param x:
    :param old_x:
    :param alpha:
    :return:
    """
    fx = alpha * x + (1 - alpha) * old_x
    return int(fx)


def control_mouse(endpoint, img):
    """Control mouse cursor

    :param endpoint:
    :param img:
    :return:
    """
    x_scale = endpoint[0] / img.shape[1]
    y_scale = endpoint[1] / img.shape[0]

    x_size, y_size = pg.size()
    X = int(x_scale * x_size)
    Y = int(y_scale * y_size)

    offset = 0
    X = clamp(X, 30, x_size - offset)
    Y = clamp(Y, 350, y_size - offset)

    alpha = 0.08

    global filtered_X, filtered_Y
    filtered_X = lowpass_filter(X, filtered_X, alpha)
    filtered_Y = lowpass_filter(Y, filtered_Y, alpha)

    pg.moveTo(filtered_X, filtered_Y)

    return filtered_X, filtered_Y


def print_text_to_frame(frame, text, org=(50, 100), font_face=cv2.FONT_HERSHEY_PLAIN, font_scale=2, color=(0, 0, 255), thickness=3):
    cv2.putText(
        img=frame,
        text=text,
        org=org,
        fontFace=font_face,
        fontScale=font_scale,
        color=color,
        thickness=thickness
    )
