from __future__ import print_function, division

import numpy as np
from math import sqrt
import os
import time
import cv2
import dlib
from MNMPF import MNMPF, pupilBorder

BIG_ANS = 1000000

predictPath = os.path.dirname(__file__)
predictPath = os.path.join(predictPath, 'predictModel')
predictPath = os.path.join(
    predictPath, 'shape_predictor_68_face_landmarks.dat')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictPath)

face_cascade = cv2.CascadeClassifier(
    'haarcascades/haarcascade_frontalface_default.xml')


def detection(frame, resizeRatio=2, verbos=True):
    start = time.time()
    kernal_size = (5, 5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, _ = gray.shape
    graySmall = cv2.resize(
        gray, (width // resizeRatio, height // resizeRatio))

    if verbos:
        print("Frame reading done at: %.2fs" % (time.time() - start))

    dets = detector(graySmall)
    if verbos:
        print(dets)
        print("Face bounding done at: %.2fs" % (time.time() - start))

    for index, faces in enumerate(dets):
        left = faces.left() * resizeRatio
        right = faces.right() * resizeRatio
        top = faces.top() * resizeRatio
        bottom = faces.bottom() * resizeRatio
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)
        face = dlib.rectangle(left, top, right, bottom)
        shape = predictor(gray, face)
        # navie method
        # (leftEyeLeftBound,leftEyeRightBound,leftEyeTopBound,leftEyeBottomBound)=(BIG_ANS,0,BIG_ANS,0)
        # (rightEyeLeftBound,rightEyeRightBound,rightEyeTopBound,rightEyeBottomBound)=(BIG_ANS,0,BIG_ANS,0)

        leftSet = np.zeros((6, 2)).astype(int)
        rightSet = np.zeros((6, 2)).astype(int)
        for i in range(36, 42):
            leftSet[i-36][0] = shape.parts()[i].x
            leftSet[i-36][1] = shape.parts()[i].y
        for i in range(42, 48):
            rightSet[i-42][0] = shape.parts()[i].x
            rightSet[i-42][1] = shape.parts()[i].y

        [leftEyeLeftBound, leftEyeTopBound] = leftSet.min(axis=0)
        [leftEyeRightBound, leftEyeBottomBound] = leftSet.max(axis=0)
        [rightEyeLeftBound, rightEyeTopBound] = rightSet.min(axis=0)
        [rightEyeRightBound, rightEyeBottomBound] = rightSet.max(axis=0)

        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
        boundOffset = 5
        leftEye = gray[leftEyeTopBound - boundOffset:leftEyeBottomBound + boundOffset,
                       leftEyeLeftBound - boundOffset:leftEyeRightBound + boundOffset]
        rightEye = gray[rightEyeTopBound - boundOffset:rightEyeBottomBound + boundOffset,
                        rightEyeLeftBound - boundOffset:rightEyeRightBound + boundOffset]

        leftEye = cv2.GaussianBlur(leftEye, (3, 3), 0)
        rightEye = cv2.GaussianBlur(rightEye, (3, 3), 0)

        if verbos:
            print("Eyes bounding done at: %.2fs" % (time.time()-start))

        (leftEyeX, leftEyeY) = MNMPF(leftEye, 3, 3)
        (rightEyeX, rightEyeY) = MNMPF(rightEye, 3, 3)

        leftEyeX = leftEyeX+leftEyeLeftBound - boundOffset
        leftEyeY = leftEyeY+leftEyeTopBound - boundOffset
        rightEyeX = rightEyeX+rightEyeLeftBound - boundOffset
        rightEyeY = rightEyeY+rightEyeTopBound - boundOffset

        frame[leftEyeY, leftEyeLeftBound:leftEyeRightBound] = (
            255, 0, 255)
        frame[leftEyeTopBound:leftEyeBottomBound,
              leftEyeX] = (255, 0, 255)
        frame[rightEyeY, rightEyeLeftBound:rightEyeRightBound] = (
            255, 0, 255)
        frame[rightEyeTopBound:rightEyeBottomBound,
              rightEyeX] = (255, 0, 255)

        cv2.rectangle(frame, (leftEyeLeftBound, leftEyeTopBound),
                      (leftEyeRightBound, leftEyeBottomBound), (255, 0, 0), 1)
        cv2.rectangle(frame, (rightEyeLeftBound, rightEyeTopBound),
                      (rightEyeRightBound, rightEyeBottomBound), (255, 0, 0), 1)

        if verbos:
            print("Eyes center detection done at: %.2fs" %
                  (time.time() - start))
            print("-----------------")

        centerLeft = leftSet.mean(axis=0)
        centerRight = rightSet.mean(axis=0)
        eyeLeft = np.array((leftEyeX, leftEyeY))
        eyeRight = np.array((rightEyeX, rightEyeY))
        frame = drawAttention(
            frame, centerLeft, centerRight, eyeLeft, eyeRight)

    return frame


def drawAttention(frame, centerLeft, centerRight, eyeLeft, eyeRight):
    print(centerLeft, centerRight, eyeLeft, eyeRight)
    x = centerRight - centerLeft
    d1 = eyeLeft - centerLeft
    d2 = eyeRight - centerRight

    cross = d1[0] * d2[1] - d1[1]*d2[0]
    if abs(cross) < 1e-5:  # parallel lines
        return frame

    t1 = (x[0] * d2[1] - x[1] * d2[0]) / cross
    intersection = (centerLeft + d1 * t1).astype(int)

    frame = cv2.line(frame, tuple(eyeLeft), tuple(
        intersection), (255, 255, 255))
    frame = cv2.line(frame, tuple(eyeRight), tuple(
        intersection), (255, 255, 255))
    frame = cv2.circle(frame, tuple(
        intersection), 3, (255, 255, 255), -1)
    return frame


def realtime_test():
    cap = cv2.VideoCapture(0)
    while(1):
        _, frame = cap.read()
        frame = detection(frame)
        cv2.imshow("capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def image_test():
    l = os.listdir('image/')
    print("start")
    print(len(l))
    for f in l:
        if f[-3:] != 'jpg':
            continue

        fname = "image/" + f
        startTime = time.time()
        frame = cv2.imread(fname)
        frame = cv2.resize(frame, (200, 267))
        frame = detection(frame)

        cv2.imwrite("image_result_dlib_faces/" + f, frame)
        # cv2.imwrite("image_result/" + "left" + f, leftEye)
        # cv2.imwrite("image_result/" + "right" + f, rightEye)
        print("Time used: %.2fs" % (time.time() - startTime))

    print("over")


def main():
    image_test()
    # realtime_test()

    # print(predictPath)


if __name__ == "__main__":
    main()
