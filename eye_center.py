import numpy as np
from math import sqrt, pow
import os
import cv2
import time
from MNMPF import MNMPF

face_cascade = cv2.CascadeClassifier(
    'haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

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
    d = d/(len(grad_x)*len(grad_x[0]))
    return d


def realtime_test():
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

            if len(eyes) > 2:
                area = np.array(map(lambda x: x[2] * x[3], eyes))
                top1 = eyes[np.argmax(np.argsort(area) == 0)]
                top2 = eyes[np.argmax(np.argsort(area) == 1)]
                # for (ex, ey, ew, eh) in [top1, top2]:
                #     cv2.rectangle(frame, (x+ex, y+ey),
                #                   (x + ex + ew, y + ey + eh), (0, 0, 255), 2)

                for (ex, ey, ew, eh) in [top1, top2]:
                    x1, y1, x2, y2 = x + \
                        int(w / 6) + ex, int(y + h / 4 + ey), x + \
                        int(w / 6) + ex + ew, int(y + h / 4 + ey + eh)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    eye = gray[y1:y2, x1:x2]
                    center = MNMPF(eye, 5, 5)
                    centerX = x1+center[0]
                    centerY = y1 + center[1]

                    cv2.circle(frame, (centerX, centerY), 3, (0, 255, 0), 2)
                    frame[centerY, :] = (128, 0, 0)
                    frame[:, centerX] = (128, 0, 0)

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
    # realtime_test()
    image_test()


if __name__ == '__main__':
    main()
