import cv2
from libdetection import detection
import multiprocessing
import time

cores=multiprocessing.cpu_count()
pool=multiprocessing.Pool(processes=cores)

def videoFromIphone():
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
    videoWriter.release

def main():
    videoFromIphone()

if __name__=="__main__":
    main()