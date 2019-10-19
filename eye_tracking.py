import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2
import numpy


def main():
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

	# img = cv2.imread("download.png")
	# gray_picture = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#make picture gray
	# faces = face_cascade.detectMultiScale(gray_picture, 1.3, 5)
	# for (x,y,w,h) in faces:
	#     cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)

	# gray_face = gray_picture[y:y+h, x:x+w] # cut the gray face frame out
	# face = img[y:y+h, x:x+w] # cut the face frame out
	# eyes = eye_cascade.detectMultiScale(gray_face)
	# for (ex,ey,ew,eh) in eyes: 
	#     cv2.rectangle(face,(ex,ey),(ex+ew,ey+eh),(0,225,255),2)

	# cv2.imshow('my image',img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	cap = cv2.VideoCapture(0)
	while(cap.isOpened()):
		bool_var, img = cap.read()
		gray_picture = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#make picture gray
		faces = face_cascade.detectMultiScale(gray_picture, 1.3, 5)
		
		for (x,y,w,h) in faces:
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
			gray_face_upper_left = gray_picture[y:y+int(h/2), x:x+int(w/2)] # cut the gray face frame out
			face_upper_left = img[y:y+int(h/2), x:x+int(w/2)]# cut the face frame out
			left_eye = eye_cascade.detectMultiScale(gray_face_upper_left)
			for (ex,ey,ew,eh) in left_eye: 
			    cv2.rectangle(face_upper_left,(ex,ey),(ex+ew,ey+eh),(0,225,255),2)

			# gray_face_upper_right = gray_picture[y:y+int(h/2), x+int(w/2):x+w] # cut the gray face frame out			
			# face_upper_right = img[y:y+int(h/2), x+int(w/2):x+w]
			# right_eye = eye_cascade.detectMultiScale(gray_face_upper_right)
			# for (ex,ey,ew,eh) in right_eye: 
			#     cv2.rectangle(face_upper_right,(ex,ey),(ex+ew,ey+eh),(0,225,255),2)

		cv2.imshow('result', img)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()