import numpy as np
from math import sqrt
import os
import time
import cv2
import dlib
from MNMPF import MNMPF,pupilBorder

BIG_ANS = 1000000

predictPath=os.path.dirname(__file__)
predictPath=os.path.join(predictPath,'predictModel')
predictPath=os.path.join(predictPath,'shape_predictor_68_face_landmarks.dat')

detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor(predictPath)

def realtime_test():
    cap = cv2.VideoCapture(0)
    kernal_size = (5, 5)

    while(1):
        # get a frame
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets=detector(gray,1)

        for index,faces in enumerate(dets):
            left=faces.left()
            right=faces.right()
            top=faces.top()
            bottom=faces.bottom()
            cv2.rectangle(frame,(left,top),(right,bottom),(0,255,0),1)
            shape=predictor(gray,faces)

            for index,faces in enumerate(dets):
                left=faces.left()
                right=faces.right()
                top=faces.top()
                bottom=faces.bottom()
                cv2.rectangle(frame,(left,top),(right,bottom),(0,255,0),1)
                shape=predictor(gray,faces)
                # navie method 
                #(leftEyeLeftBound,leftEyeRightBound,leftEyeTopBound,leftEyeBottomBound)=(BIG_ANS,0,BIG_ANS,0)
                #(rightEyeLeftBound,rightEyeRightBound,rightEyeTopBound,rightEyeBottomBound)=(BIG_ANS,0,BIG_ANS,0)
                
                leftSet=np.zeros((6,2)).astype(int)
                rightSet=np.zeros((6,2)).astype(int)
                for i in range(36,42):
                    leftSet[i-36][0]=shape.parts()[i].x
                    leftSet[i-36][1]=shape.parts()[i].y
                for i in range(42,48):
                    rightSet[i-42][0]=shape.parts()[i].x
                    rightSet[i-42][1]=shape.parts()[i].y
                    
                [leftEyeLeftBound,leftEyeTopBound]=leftSet.min(axis=0)
                [leftEyeRightBound,leftEyeBottomBound]=leftSet.max(axis=0)
                [rightEyeLeftBound,rightEyeTopBound]=rightSet.min(axis=0)
                [rightEyeRightBound,rightEyeBottomBound]=rightSet.max(axis=0)

                gray=cv2.cvtColor(gray,cv2.COLOR_RGB2GRAY)
                leftEye=gray[leftEyeTopBound:leftEyeBottomBound,leftEyeLeftBound:leftEyeRightBound]
                rightEye=gray[rightEyeTopBound:rightEyeBottomBound,rightEyeLeftBound:rightEyeRightBound]

                leftEye=cv2.GaussianBlur(leftEye,(3,3),0)
                rightEye=cv2.GaussianBlur(rightEye,(3,3),0)

                (leftCenterX,leftCenterY)=MNMPF(leftEye,3,3)
                (rightCenterX,rightCenterY)=MNMPF(rightEye,3,3)

                leftCenterX=leftCenterX+leftEyeLeftBound
                leftCenterY=leftCenterY+leftEyeTopBound
                rightCenterX=rightCenterX+rightEyeLeftBound
                rightCenterY=rightCenterY+rightEyeTopBound

                frame[leftCenterY,leftEyeLeftBound:leftEyeRightBound]=(255,0,255)
                frame[leftEyeTopBound:leftEyeBottomBound,leftCenterX]=(255,0,255)
                frame[rightCenterY,rightEyeLeftBound:rightEyeRightBound]=(255,0,255)
                frame[rightEyeTopBound:rightEyeBottomBound,rightCenterX]=(255,0,255)

        # show a frame
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
        # print(fname)
        frame = cv2.imread(fname)
        frame = cv2.resize(frame, (200, 267))

        start_time = time.time()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # gray = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        dets=detector(gray,1)
        #print("{num} faces detected".format(num=len(dets)))
        # 36-41 index left side
        # 42-47 index right side
        for index,faces in enumerate(dets):
            left=faces.left()
            right=faces.right()
            top=faces.top()
            bottom=faces.bottom()
            cv2.rectangle(frame,(left,top),(right,bottom),(0,255,0),1)
            shape=predictor(gray,faces)
            # navie method 
            #(leftEyeLeftBound,leftEyeRightBound,leftEyeTopBound,leftEyeBottomBound)=(BIG_ANS,0,BIG_ANS,0)
            #(rightEyeLeftBound,rightEyeRightBound,rightEyeTopBound,rightEyeBottomBound)=(BIG_ANS,0,BIG_ANS,0)
            
            leftSet=np.zeros((6,2)).astype(int)
            rightSet=np.zeros((6,2)).astype(int)
            for i in range(36,42):
                leftSet[i-36][0]=shape.parts()[i].x
                leftSet[i-36][1]=shape.parts()[i].y
            for i in range(42,48):
                rightSet[i-42][0]=shape.parts()[i].x
                rightSet[i-42][1]=shape.parts()[i].y
                
            [leftEyeLeftBound,leftEyeTopBound]=leftSet.min(axis=0)
            [leftEyeRightBound,leftEyeBottomBound]=leftSet.max(axis=0)
            [rightEyeLeftBound,rightEyeTopBound]=rightSet.min(axis=0)
            [rightEyeRightBound,rightEyeBottomBound]=rightSet.max(axis=0)

            

            '''for index,points in enumerate(shape.parts()):
               
                if index in range(36,42):
                    if points.x<leftEyeLeftBound:
                        leftEyeLeftBound=points.x
                    if points.x>leftEyeRightBound:
                        leftEyeRightBound=points.x
                    if points.y<leftEyeTopBound:
                        leftEyeTopBound=points.y
                    if points.y>leftEyeBottomBound:
                        leftEyeBottomBound=points.y
                elif index in range(42,48):
                    if points.x<rightEyeLeftBound:
                        rightEyeLeftBound=points.x
                    if points.x>rightEyeRightBound:
                        rightEyeRightBound=points.x
                    if points.y<rightEyeTopBound:
                        rightEyeTopBound=points.y
                    if points.y>rightEyeBottomBound:
                        rightEyeBottomBound=points.y'''
            gray=cv2.cvtColor(gray,cv2.COLOR_RGB2GRAY)
            leftEye=gray[leftEyeTopBound:leftEyeBottomBound,leftEyeLeftBound:leftEyeRightBound]
            rightEye=gray[rightEyeTopBound:rightEyeBottomBound,rightEyeLeftBound:rightEyeRightBound]

            leftEye=cv2.GaussianBlur(leftEye,(3,3),0)
            rightEye=cv2.GaussianBlur(rightEye,(3,3),0)

            (a,b,c,d)=pupilBorder(leftEye)
            (m,n,p,q)=pupilBorder(rightEye)

            (leftCenterX,leftCenterY)=MNMPF(leftEye[b:d,a:c],3,3)
            (rightCenterX,rightCenterY)=MNMPF(rightEye[n:q,m:p],3,3)


            '''(leftCenterX,leftCenterY)=MNMPF(leftEye,3,3)
            (rightCenterX,rightCenterY)=MNMPF(rightEye,3,3)'''
            
            leftCenterX=leftCenterX+a
            leftCenterY=leftCenterY+b
            rightCenterX=rightCenterX+m
            rightCenterX=rightCenterX+n

            leftEye[leftCenterY][leftCenterX]=255
            rightEye[rightCenterY][rightCenterX]=255

            leftCenterX=leftCenterX+leftEyeLeftBound
            leftCenterY=leftCenterY+leftEyeTopBound
            rightCenterX=rightCenterX+rightEyeLeftBound
            rightCenterY=rightCenterY+rightEyeTopBound

            frame[leftCenterY,leftEyeLeftBound:leftEyeRightBound]=(255,0,255)
            frame[leftEyeTopBound:leftEyeBottomBound,leftCenterX]=(255,0,255)
            frame[rightCenterY,rightEyeLeftBound:rightEyeRightBound]=(255,0,255)
            frame[rightEyeTopBound:rightEyeBottomBound,rightCenterX]=(255,0,255)

            '''leftEye[leftCenterY,:]=128
            leftEye[:,leftCenterX]=128
            rightEye[rightCenterY,:]=128
            rightEye[:,rightCenterX]=128'''

            cv2.rectangle(frame,(leftEyeLeftBound,leftEyeTopBound),(leftEyeRightBound,leftEyeBottomBound),(255,0,0),1)
            cv2.rectangle(frame,(rightEyeLeftBound,rightEyeTopBound),(rightEyeRightBound,rightEyeBottomBound),(255,0,0),1)

        cv2.imwrite("image_result_dlib_faces/" + f, frame)
        cv2.imwrite("image_result/" + "left" + f, leftEye)
        cv2.imwrite("image_result/" + "right" + f, rightEye)
        print("Time used: %.2fs" % (time.time() - start_time))

    print("over")

def main():
    #image_test()
    realtime_test()

    #print(predictPath)

if __name__=="__main__":
    main()