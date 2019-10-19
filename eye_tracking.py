import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

# import cv2
# import numpy


# def main():
# 	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# 	eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
# 	detector_params = cv2.SimpleBlobDetector_Params()
# 	detector_params.filterByArea = True
# 	detector_params.maxArea = 1500
# 	detector = cv2.SimpleBlobDetector_create(detector_params)

# 	# img = cv2.imread("download.png")
# 	# gray_picture = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#make picture gray
# 	# faces = face_cascade.detectMultiScale(gray_picture, 1.3, 5)
# 	# for (x,y,w,h) in faces:
# 	#     cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)

# 	# gray_face = gray_picture[y:y+h, x:x+w] # cut the gray face frame out
# 	# face = img[y:y+h, x:x+w] # cut the face frame out
# 	# eyes = eye_cascade.detectMultiScale(gray_face)
# 	# for (ex,ey,ew,eh) in eyes: 
# 	#     cv2.rectangle(face,(ex,ey),(ex+ew,ey+eh),(0,225,255),2)

# 	# cv2.imshow('my image',img)
# 	# cv2.waitKey(0)
# 	# cv2.destroyAllWindows()

# 	cap = cv2.VideoCapture(0)
# 	while(cap.isOpened()):
# 		bool_var, img = cap.read()
# 		img = cv2.flip(img, 1)
# 		gray_picture = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#make picture gray
# 		faces = face_cascade.detectMultiScale(gray_picture, 1.3, 5)
		
# 		for (x,y,w,h) in faces:
# 			cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
# 			gray_face_upper_left = gray_picture[y:y+int(h/2), x:x+int(w/2)] # cut the gray face frame out
			
# 			face_upper_left = img[y:y+int(h/2), x:x+int(w/2)]# cut the face frame out
# 			left_eye = eye_cascade.detectMultiScale(gray_face_upper_left)
# 			for (ex,ey,ew,eh) in left_eye: 
# 			    cv2.rectangle(face_upper_left,(ex,ey),(ex+ew,ey+eh),(0,225,255),2)

# 			# gray_face_upper_right = gray_picture[y:y+int(h/2), x+int(w/2):x+w] # cut the gray face frame out			
# 			# face_upper_right = img[y:y+int(h/2), x+int(w/2):x+w]
# 			# right_eye = eye_cascade.detectMultiScale(gray_face_upper_right)
# 			# for (ex,ey,ew,eh) in right_eye: 
# 			#     cv2.rectangle(face_upper_right,(ex,ey),(ex+ew,ey+eh),(0,225,255),2)

# 		cv2.imshow('result', img)
# 		if cv2.waitKey(1) & 0xFF == ord('q'):
# 			break
# 	cap.release()
# 	cv2.destroyAllWindows()


# if __name__ == "__main__":
# 	main()

import cv2
import numpy as np


# init part
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector = cv2.SimpleBlobDetector_create(detector_params)


def detect_faces(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(coords) > 1:
        biggest = (0, 0, 0, 0)
        for i in coords:
            if i[3] > biggest[3]:
                biggest = i
        biggest = np.array([i], np.int32)
    elif len(coords) == 1:
        biggest = coords
    else:
        return None
    for (x, y, w, h) in biggest:
        frame = img[y:y + h, x:x + w]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
    return frame


def detect_eyes(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = cascade.detectMultiScale(gray_frame, 1.3, 5)  # detect eyes
    width = np.size(img, 1)  # get face frame width
    height = np.size(img, 0)  # get face frame height
    left_eye = None
    right_eye = None
    for (x, y, w, h) in eyes:
        if y > height / 2:
            pass
        eyecenter = x + w / 2  # get the eye center
        if eyecenter < width * 0.5:
            left_eye = img[y:y + h, x:x + w]
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        else:
            right_eye = img[y:y + h, x:x + w]
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)

    return left_eye, right_eye


def cut_eyebrows(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    img = img[eyebrow_h:height, 0:width]  # cut eyebrows out (15 px)
    return img


def blob_process(img, threshold, detector):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, None, iterations=2)
    img = cv2.dilate(img, None, iterations=4)
    img = cv2.medianBlur(img, 5)
    keypoints = detector.detect(img)
    print(keypoints)
    return keypoints


def nothing(x):
    pass


def main():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('image')
    cv2.createTrackbar('threshold', 'image', 0, 255, nothing)
    while True:
        _, frame = cap.read()
        face_frame = detect_faces(frame, face_cascade)
        if face_frame is not None:
            eyes = detect_eyes(face_frame, eye_cascade)
            # for eye in eyes:
            #     if eye is not None:
            #         # threshold = r = cv2.getTrackbarPos('threshold', 'image')
            #         threshold = 79
            #         eye = cut_eyebrows(eye)
            #         keypoints = blob_process(eye, threshold, detector)
            #         # eye = cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()