import cv2
from libdetection import detection
import multiprocessing
import time
import numpy as np

cores=multiprocessing.cpu_count()
pool=multiprocessing.Pool(processes=cores)

def realTime():
    videoCapture=cv2.VideoCapture(0)
    success,frame = videoCapture.read()
    frameSet=[]
    count=0
    while success:
        newframe=frame
        frameSet.append(newframe)
        if count==8:
            ansSet=pool.map(detection,frameSet)
            frameSet=[]
            count=0
            intersectionSet=np.zeros(shape=(8,2))

            for i in range(8):
                intersectionSet[i][0]=ansSet[i][1][0]
                intersectionSet[i][1]=ansSet[i][1][1]
            xM=np.mean(intersectionSet,axis=0).astype(int)
            meanFrame=list(ansSet[7])[0]
            meanFrame[xM[1],:]=(255,0,255)
            meanFrame[:,xM[0]]=(0,0,255)
            cv2.imshow("capture",meanFrame)
                
        count=count+1
        success, frame = videoCapture.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # get the next frame of the video


def main():
    realTime()

if __name__=="__main__":
    main()