'''
MNMPF Projection algorithm
'''

import numpy as np
from math import sqrt, pow
import cv2

BIG_ANS = 1000000

'''""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

'''""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def pupilBorder(img):
    # (height,width)=img.shape
    xProjection = np.sum(img, axis=0)
    yProjection = np.sum(img, axis=1)
    xThreshold = np.min(xProjection)*1.5
    yThreshold = np.min(yProjection)*1.5

    leftBound = np.where(
        xProjection == xProjection[xProjection < xThreshold][0])[0][0]
    rightBound = np.where(
        xProjection == xProjection[xProjection < xThreshold][-1])[0][-1]
    topBound = np.where(
        yProjection == yProjection[yProjection < yThreshold][0])[0][0]
    bottomBound = np.where(
        yProjection == yProjection[yProjection < yThreshold][-1])[0][-1]

    '''for x in range(width):
        if xProjection[x]<xThreshold:
            leftBound=x
            break
    for x in range(width):
        if xProjection[width-1-x]<xThreshold:
            rightBound=width-1-x
            break
    for y in range(height):
        if yProjection[y]<yThreshold:
            topBound=y
            break
    for y in range(height):
        if yProjection[height-1-y]<yThreshold:
            bottomBound=height-1-y
            break'''
    return (leftBound, topBound, rightBound, bottomBound)


'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''


def areaMean(img, x, y, rx, ry):
    '''calculate the img's mean grayscale inside a bounding box'''
    (height, width) = img.shape
    img_bounded = img[max(0, y-ry):min(height, y+ry+1),
                      max(0, x-rx):min(width, x+rx+1)]
    return np.mean(img_bounded)


def projectionX(img, x0, rx, ry):
    '''return the min areaMean value of every horizon lines'''
    proj = np.vectorize(lambda i: areaMean(img, x0, i, rx, ry))
    area_mean_of_lines = proj(np.arange(img.shape[0]))

    minY = np.argmin(area_mean_of_lines)
    minMean = np.min(area_mean_of_lines)
    return (minMean, minY)

# same result? redundent ?


def projectionY(img, y0, rx, ry):
    '''return the min areaMean value of every vertical lines'''
    proj = np.vectorize(lambda i: areaMean(img, i, y0, rx, ry))
    area_mean_of_lines = proj(np.arange(img.shape[1]))

    minY = np.argmin(area_mean_of_lines)
    minMean = np.min(area_mean_of_lines)
    return (minMean, minY)


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
