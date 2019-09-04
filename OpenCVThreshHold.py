import cv2

import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
def ShowAmplitudeBar(ListAmplitude):
    for i, Amplitude in enumerate(ListAmplitude):
        plt.bar(x=i, height=Amplitude, align='center', width=1)
    plt.show()
RealImage = cv2.imread('Capture.png')
# img = cv2.imread('logo.jpg')
grayImage = cv2.cvtColor(RealImage, cv2.COLOR_BGR2GRAY)
_, thresh1 = cv2.threshold(grayImage, 175, 255, cv2.THRESH_BINARY)
# print(thresh1.shape)
# cv2.imshow('HH', thresh1)
ListAmplitude = []
for i in range(thresh1.shape[0]):
    Amplitude = 0
    for ii in range(thresh1.shape[1]):
        if thresh1[i, ii] == 0:
            Amplitude += 1
    ListAmplitude.append(Amplitude)

SpaceWidth  = 5

CurrentAmplitude = 0
ListCroppedLine = []
ListCroppedRealLine = []
StartPoint = 0
EndPoint = 0
for i, Amplitude in enumerate(ListAmplitude):
    if Amplitude > 0 and CurrentAmplitude == 0:
        StartPoint = i
    if Amplitude == 0 and CurrentAmplitude > 0:
        EndPoint = i
        ListCroppedLine.append(thresh1[max(StartPoint-5,0):min(EndPoint+5,thresh1.shape[0]), :])
        ListCroppedRealLine.append(RealImage[max(StartPoint-5,0):min(EndPoint+5,thresh1.shape[0]), :])
    CurrentAmplitude = Amplitude

for IndexList in range(len(ListCroppedLine)):
    img =  ListCroppedLine[IndexList]
    imgReal = ListCroppedRealLine[IndexList]
    cv2.imwrite('./Result/Line.png',img)
    ListAmplitude=[]
    ListCroppedWord = []
    ListCroppedRealWord =[]

    for i in range(img.shape[1]):
        Amplitude = 0
        for ii in range(img.shape[0]):
            if img[ii,i] == 0:
                Amplitude += 1
        ListAmplitude.append(Amplitude)
    StartPoint = 0
    EndPoint = 0
    IsDetectEndPoint = True
    for i, Amplitude in enumerate(ListAmplitude):
        if Amplitude > 0 and CurrentAmplitude == 0 and IsDetectEndPoint:
            StartPoint = i
            IsDetectEndPoint = False
        if Amplitude == 0 and CurrentAmplitude > 0:
            IsCoupleOfPixelABlack = False
            for ii in range(SpaceWidth):
                if i+ii<len(ListAmplitude) and ListAmplitude[i+ii]>0:
                    IsCoupleOfPixelABlack = True
            if not IsCoupleOfPixelABlack:
                IsDetectEndPoint = True
                EndPoint = i
                print('StartPoint: {}, EndPoint: {}'.format(StartPoint,EndPoint))
                ListCroppedWord.append(img[:,max(StartPoint,0):min(EndPoint,thresh1.shape[1])])
                ListCroppedRealWord.append(imgReal[:,max(StartPoint,0):min(EndPoint,thresh1.shape[1])])
        CurrentAmplitude = Amplitude

    for i, Img in enumerate(ListCroppedRealWord):
        cv2.imwrite('./Result/Line{}Word{}.png'.format(IndexList,i),Img)


# ShowAmplitudeBar(ListAmplitude)
