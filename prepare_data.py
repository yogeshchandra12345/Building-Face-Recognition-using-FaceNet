"""
Steps to follow for making a face recognition system from scratch.
1. Ready DataSets. Samples of human faces which you want to recognize.
2. Training Model.  Train an existing model with this new dataset.
3. Prediction using Trained Model.
"""
import cv2
import numpy as np

# import a trained model to detect any human face. ( Using OpenCV harcascade classifier for human face detection.)

face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')


def face_extractor(img):
	"""
	Extract the face feature from image
	"""

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_classifier.detectMultiScale(gray, 1.3, 5) # scaling factor 1.3, minimum neighbour 5.

	if faces is ():
		return None

	for (x,y,w,h) in faces:
		cropped_faces = img[y:y+h, x:x+w]
	return cropped_faces

cap = cv2.VideoCapture(0)
print('capturing face')

count = 1

while True:
	ret, frame = cap.read()
	if face_extractor(frame) is not None:
		count += 1
		face = cv2.resize(face_extractor(frame), (200,200))
		face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

		file_name_path = './faces/user{}.jpg'.format(str(count))
		cv2.imwrite(file_name_path, face)

		cv2.putText(face, str(count), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

		cv2.imshow('Face Cropper', face)
	else:
		print('Face not found')
		pass

	if cv2.waitKey(1)==13 or count == 500: # if enter key is presses, 13 is ASCII code
		break

cap.release()
cv2.destroyAllWindows()
print('Collecting Samples Complete !!!')

