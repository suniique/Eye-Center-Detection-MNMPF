'''
MNMPF Projection algorithm
'''

import numpy as np
from math import sqrt, pow
import cv2

BIG_ANS = 1000000


def areaMean(img, x, y, rx, ry):
    '''claculate the img's mean grayscale inside a bounding box'''
    (height, width) = img.shape
    img_bounded = img[max(0, y-ry):min(height, y+ry+1),
                      max(0, x-rx):min(width, x+rx+1)]
    return np.mean(img_bounded)


def projectionX(img, x0, rx, ry):
    '''return the min areaMean value of every horizon lines'''
    (height, width) = img.shape
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
    '''return the min areaMean value of every vertical lines'''
    (height, width) = img.shape
    minMean = areaMean(img, 0, y0, rx, ry)
    minX = 0
    for i in range(width):
        tempMin = areaMean(img, i, y0, rx, ry)
        if minMean > tempMin:
            minMean = tempMin
            minX = i
    return (minMean, minX)


def MNMPF(img, rx, ry):
    (height, width) = img.shape
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
