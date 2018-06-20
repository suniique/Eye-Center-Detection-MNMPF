import numpy as np
from math import sqrt, pow
import os
import cv2
import time

face_cascade = cv2.CascadeClassifier(
    'haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')


# img has to be grayscale

# projection algorithm

BIG_ANS = 1000000

'''""""""""""""""""""""""""""""""""""""""""""""""""""""""
start of algorithm
""""""""""""""""""""""""""""""""""""""""""""""""""""""'''


def areaMean(img, x, y, rx, ry):
    mny = 0
    height = len(img)
    width = len(img[0])
    for dy in range(-ry, ry+1):
        for dx in range(-rx, rx+1):
            if y+dy < 0 or x+dx < 0 or y+dy >= height or x+dx >= width:
                return BIG_ANS
            mny = img[y+dy][x+dx]+mny
    areaSize = (rx*2+1)*(ry*2+1)
    return mny/areaSize


def projectionX(img, x0, rx, ry):
    height = len(img)
    minMean = areaMean(img, x0, 0, rx, ry)
    minY = 0
    for i in range(height):
        tempMin = areaMean(img, x0, i, rx, ry)
        if minMean > tempMin:
            minMean = tempMin
            minY = i
    return (minMean, minY)

# same result? redundent ?


def projectionY(img, y0, rx, ry):
    width = len(img[0])
    minMean = areaMean(img, 0, y0, rx, ry)
    minX = 0
    for i in range(width):
        tempMin = areaMean(img, i, y0, rx, ry)
        if minMean > tempMin:
            minMean = tempMin
            minX = i
    return (minMean, minX)


def MNMPF(img, rx, ry):
    width = len(img[0])
    height = len(img)
    centerX = 0
    centerY = 0
    (minProjectioX, minY) = projectionX(img, 0, rx, ry)
    (minProjectionY, minX) = projectionY(img, 0, rx, ry)
    for px in range(width):
        (tempMinProjectionX, tempMinY) = projectionX(img, px, rx, ry)
        if minProjectioX > tempMinProjectionX:
            minProjectioX = tempMinProjectionX
            centerX = px
            minY = tempMinY
    for py in range(height):
        (tempMinProjectionY, tempMinX) = projectionY(img, py, rx, ry)
        if minProjectionY > tempMinProjectionY:
            minProjectionY = tempMinProjectionY
            centerY = py
            minX = tempMinX
    return (centerX, centerY)


'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""
end of algorithm
"""""""""""""""""""""""""""""""""""""""""""""""""""""""'''

# image gradient algorithm


def eye_center(img, grad_x, grad_y):
    center_x = 0
    center_y = 0
    max_c = 0
    for j in range(len(img)):
        for i in range(len(img[0])):
            c = c_compute(img, j, i, grad_x, grad_y)*img[j][i]
            if c > max_c:
                max_c = c
                center_y = j
                center_x = i
            else:
                continue
    return [center_y, center_x]

# image gradient algorithm


def c_compute(img, point_y, point_x, grad_x, grad_y):
    d = 0
    for n in range(len(grad_y)):
        for m in range(len(grad_x)):
            dx = m-point_x
            dy = n-point_y
            if dx == 0 and dy == 0:
                continue
            magnitude = sqrt(dx**2+dy**2)
            dx = dx/magnitude
            dy = dy/magnitude
            grad_mag = sqrt((grad_x[n][m])**2+(grad_y[n][m])**2)
            # print(grad_mag)
            if(grad_mag == 0):
                continue
            d = d+((grad_x[n][m]/grad_mag*dx)+(grad_y[n][m]/grad_mag*dy))**2
            # print(((dx*grad_x[n][m])+(dy*grad_y[n][m])))
    # print point_y
    # print point_x
    # print(d)
    d = d/(len(grad_x)*len(grad_x[0]))
    return d


def realtime():
    cap = cv2.VideoCapture(0)
    kernal_size = (5, 5)

    while(1):
        # get a frame
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, kernal_size, 0)
        faces = face_cascade.detectMultiScale(gray)

        max_face = None

        if len(faces) > 0:
            if len(faces) > 1:
                max_area = 0
                for (x, y, w, h) in faces:
                    area = w * h
                    if area > max_area:
                        max_area = area
                        max_face = (x, y, w, h)
            elif len(faces) == 1:
                max_face = faces[0]
            (x, y, w, h) = max_face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            roi_gray = gray[int(y + h / 4):int(y + h / 1.7),
                            x+int(w/6):x + int(5*w/6)]
            cv2.rectangle(frame, (x+int(w/6), int(y + h / 5)),
                          (x + int(5*h/6), int(y + h / 2)), (0, 255, 0), 2)
            eyes = eye_cascade.detectMultiScale(roi_gray)

            # if len(eyes) > 2:
            #     area = np.array(map(lambda x: x[2] * x[3], eyes))

            #     top1 = eyes[np.argmax(np.argsort(area) == 0)]
            #     top2 = eyes[np.argmax(np.argsort(area) == 1)]

            #     for (ex, ey, ew, eh) in [top1, top2]:
            #         cv2.rectangle(frame, (x+ex, y+ey),
            #                       (x + ex + ew, y + ey + eh), (0, 0, 255), 2)
            # else:
            for (ex, ey, ew, eh) in eyes:
                #print("{ex} {ey} {ew} {eh}".format(ex=ex,ey=ey,ew=ew,eh=eh))
                x1, y1, x2, y2 = x + \
                    int(w / 6) + ex, int(y + h / 4 + ey), x + \
                    int(w / 6) + ex + ew, int(y + h / 4 + ey + eh)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                eye = gray[y1:y2, x1:x2]
                center = MNMPF(eye, 5, 5)
                print(center)
                centerX = x1+center[0]
                centerY = y1+center[1]
                cv2.circle(frame, (centerX, centerY), 3, (0, 255, 0), 2)
                # (centerX,centerY)=MNMPF(im)

                '''eye=gray[y1:y2,x1:x2]
                gradx=cv2.Sobel(eye,cv2.CV_16S,1,0,ksize=3)
                grady=cv2.Sobel(eye,cv2.CV_16S,0,1,ksize=3)
                #print(np.array(eye).shape)
                center=eye_center(eye,gradx,grady)
                eye_y=center[0]+y1
                eye_x=center[1]+x1
                cv2.circle(frame,(eye_x,eye_y),2,(0,255,255),3)'''

        # show a frame
        cv2.imshow("capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def image_test():
    l = os.listdir('image/')
    print("start")
    for f in l:
        if f[-3:] != 'jpg':
            continue

        fname = "image/" + f
        # print(fname)
        frame = cv2.imread(fname)
        frame = cv2.resize(frame, (200, 267))

        start_time = time.time()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        faces = face_cascade.detectMultiScale(gray)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+h, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+h]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey),
                              (ex+ew, ey+eh), (0, 255, 0), 2)
                roi_eye = cv2.GaussianBlur(
                    roi_gray[ey:ey+eh, ex:ex+ew], (3, 3), 0)
                (centerX, centerY) = MNMPF(roi_eye, 3, 3)
                roi_eye[centerY][centerX] = 255
                roi_eye[centerY, :] = 128
                roi_eye[:, centerX] = 128
                # cv2.circle(roi_eye,(centerX,centerY),1,(255,0,0),4)
                #roi_eye=roi_gray[ey:ey+eh, ex:ex+ew]

                '''new_ones = np.ones((len(roi_eye), len(roi_eye[0])))
                roi_eye1 = new_ones-roi_eye
                grad = np.gradient(roi_eye1)
                grad_x = grad[1]
                grad_y = grad[0]
                d = eye_center(roi_eye1, grad_x, grad_y)
                cv2.circle(roi_color, (d[1]+ex, d[0]+ey), 2, (255, 0, 0), 4)'''
        # img has to be a grayscale

        cv2.imwrite("image_result/" + f, roi_eye)
        #print("saved roi: " + f)
        # cv2.imshow("roi_color", roi_color)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    print("Time used: %.2fs" % (time.time() - start_time))
    print("over")


def main():
    # realtime()
    image_test()


if __name__ == '__main__':
    main()
