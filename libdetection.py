from __future__ import print_function, division

import numpy as np
from math import sqrt
import os
import time
import cv2
import dlib
from MNMPF import MNMPF, pupilBorder
from vec3d import midPoint, zMidPoint
import random
import multiprocessing

thick = 1
eyeR = 100
H = 2000
fontSize=0.5
cores=multiprocessing.cpu_count()
pool=multiprocessing.Pool(processes=cores)

predictPath = os.path.dirname(__file__)
predictPath = os.path.join(predictPath, 'predictModel')
predictPath = os.path.join(
    predictPath, 'shape_predictor_68_face_landmarks.dat')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictPath)

face_cascade = cv2.CascadeClassifier(
    'haarcascades/haarcascade_frontalface_default.xml')


def detection(frame, resizeRatio=1, verbos=False):
    start = time.time()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, _ = gray.shape
    graySmall = cv2.resize(
        gray, (width // resizeRatio, height // resizeRatio))

    if verbos:
        print("Frame reading done at: %.2fs" % (time.time() - start))

    dets = detector(graySmall)
    if verbos:
        print("Face bounding done at: %.2fs" % (time.time() - start))

    for faces in dets:
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
        #print(leftEyeTopBound-leftEyeBottomBound)
        #print(rightEyeTopBound-rightEyeBottomBound)

        # gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
        gray = np.mean(gray[..., [0]], axis=2).astype(np.uint8)
        gray = cv2.equalizeHist(gray)
        if verbos:
            print(gray.shape)

        boundOffset = 5
        leftEye = gray[leftEyeTopBound - boundOffset:leftEyeBottomBound + boundOffset,
                       leftEyeLeftBound - boundOffset:leftEyeRightBound + boundOffset]
        rightEye = gray[rightEyeTopBound - boundOffset:rightEyeBottomBound + boundOffset,
                        rightEyeLeftBound - boundOffset:rightEyeRightBound + boundOffset]

        leftEye = cv2.GaussianBlur(leftEye, (3, 3), 0)
        rightEye = cv2.GaussianBlur(rightEye, (3, 3), 0)

        '''""""""""""""""""""""""""""""""""""""
        '''""""""""""""""""""""""""""""""""""""
        '''(a,b,c,d)=pupilBorder(leftEye)
        (m,n,p,q)=pupilBorder(rightEye)

        (leftEyeX,leftEyeY)=MNMPF(leftEye[b:d,a:c],3,3)
        (rightEyeX,rightEyeY)=MNMPF(rightEye[n:q,m:p],3,3)'''
        # cv2.imshow('Eye', leftEye)
        '''""""""""""""""""""""""""""""""""""""
        '''""""""""""""""""""""""""""""""""""""

        if verbos:
            print("Eyes bounding done at: %.2fs" % (time.time()-start))

        (leftEyeX, leftEyeY) = MNMPF(leftEye, 3, 3)
        (rightEyeX, rightEyeY) = MNMPF(rightEye, 3, 3)

        #use pupilBorder
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
        centerLeft = leftSet.mean(axis=0)
        centerRight = rightSet.mean(axis=0)

        #centerLeft=[(leftEyeLeftBound+leftEyeRightBound)/2,(leftEyeTopBound+leftEyeBottomBound)/2]
        #centerRight=[(rightEyeLeftBound+rightEyeRightBound)/2,(rightEyeTopBound+rightEyeBottomBound)/2]

        eyeLeft = np.array((leftEyeX, leftEyeY))
        eyeRight = np.array((rightEyeX, rightEyeY))

        frame,intersect = drawAttention(frame, centerLeft, centerRight, eyeLeft, eyeRight)
        if verbos:
            print("intersection detection done at: %.2fs" %
                  (time.time() - start))
            print("-----------------")
        return frame,intersect
    return frame,(350,700)

def drawAttention(frame, centerLeft, centerRight, eyeLeft, eyeRight):
    O1 = np.array([centerLeft[0], centerLeft[1], 0])
    O2 = np.array([centerRight[0], centerRight[1], 0])
    
    dy1=eyeLeft[0] - centerLeft[0]
    dx1=eyeLeft[1] - centerLeft[1]
    dz1=sqrt(eyeR**2-dy1**2-dx1**2)

    dy2=eyeRight[0] - centerRight[0]
    dx2=eyeRight[1] - centerRight[1]
    dz2=sqrt(eyeR**2-dy2**2-dx2**2)

    W1 = np.array([dy1,
                   dx1, dz1])
    W2 = np.array([dy2 ,
                   dx2, dz2])

    P = midPoint(O1, W1, O2, W2).astype(int)
    zPoint = zMidPoint(O1, W1, O2, W2, H).astype(int)

    x = centerRight - centerLeft
    d1 = eyeLeft - centerLeft
    d2 = eyeRight - centerRight

    cross = d1[0] * d2[1] - d1[1]*d2[0]
    if abs(cross) < 1e-5:  # parallel lines
        return frame,(350,700)

    #t1 = (x[0] * d2[1] - x[1] * d2[0]) / cross
    # intersection = (centerLeft + d1 * t1).astype(int)

    intersection1 = (P[0, 0], P[1, 0])
    intersection2 = (zPoint[0], zPoint[1])

    #frame = drawIntersection(frame, intersection1, (255,255,0),eyeLeft, eyeRight)
    #frame = drawIntersection(frame, intersection2, (255,0,255),eyeLeft, eyeRight)
    return frame,intersection2

def drawIntersection(frame, intersection, color, eyeLeft, eyeRight):
    frame = cv2.line(frame, tuple(eyeLeft), tuple(
        intersection), color, thick)
    frame = cv2.line(frame, tuple(eyeRight), tuple(
        intersection), color, thick)
    frame = cv2.circle(frame, tuple(
        intersection), 2*thick, color, -1)
    if intersection[1] in range(len(frame)):
        frame[intersection[1],:]=(255,255,0)
    frame=cv2.putText(frame,"({},{})".format(intersection[0],intersection[1]),(intersection[0],intersection[1]),cv2.FONT_ITALIC,fontSize,(255,0,0),2)
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
        #frame = cv2.resize(frame, (200, 267))
        frame = detection(frame)

        cv2.imwrite("image_result_dlib_faces/" + f, frame)
        # cv2.imwrite("image_result/" + "left" + f, leftEye)
        # cv2.imwrite("image_result/" + "right" + f, rightEye)
        print("Time used: %.2fs" % (time.time() - startTime))
        
    print("over")

'''def videoFromIphone():
    videoCapture=cv2.VideoCapture("iphoneVideo/test1.MOV")
    fps=videoCapture.get((cv2.CAP_PROP_FPS))
    size=(int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    videoWriter = cv2.VideoWriter('iphoneVideo/answer1.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)
    success,frame = videoCapture.read()
    frameSet=[]
    count=0
    startTime=time.time()
    while success:
        newframe=cv2.transpose(frame)
        #newframe = detection(newframe)
        frameSet.append(newframe)
        if count==20:
            ansSet=pool.map(detection,frameSet)
            frameSet=[]
            count=0
            for i in range(20):
                videoWriter.write(ansSet[i])
        #cv2.imshow("capture",newframe)
        #videoWriter.write(newframe)  # write one frame into the output video
        count=count+1
        success, frame = videoCapture.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # get the next frame of the video
    print(time.time()-startTime)
    cv2.destroyAllWindows()     # close all the widows opened inside the program
    videoCapture.release        # release the video read/write handler
    videoWriter.release'''



def main():
    #image_test()
    #realtime_test()
    #videoFromIphone()
    pass


if __name__ == "__main__":
    main()
