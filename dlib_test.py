import cv2
import os
import dlib
import numpy as np
import time

predictPath = os.path.dirname(__file__)
predictPath = os.path.join(predictPath, 'predictModel')
predictPath = os.path.join(
    predictPath, 'shape_predictor_68_face_landmarks.dat')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictPath)

face_cascade = cv2.CascadeClassifier(
    'haarcascades/haarcascade_frontalface_default.xml')


def image_test():
    start = time.time()
    frame = cv2.imread('image/test3.jpg')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print("Read image time: %.2fs" % (time.time() - start))

    start = time.time()
    faces = detector(gray)
    print(len(faces))
    print(faces[0] * 2)
    print("Dlib face time: %.2fs" % (time.time() - start))
    print(faces)

    start = time.time()
    faces = face_cascade.detectMultiScale(gray)
    print("CV face time: %.2fs" % (time.time() - start))
    print(faces)


def main():
    image_test()


if __name__ == '__main__':
    main()
