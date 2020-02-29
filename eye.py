import cv2
import dlib
import numpy as np
import os
import sys


class Eye:
    def __init__(self):
        pass



if __name__ == "__main__":
    image = cv2.imread("women_face.jpg")


    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
