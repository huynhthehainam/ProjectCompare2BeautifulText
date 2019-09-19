#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 13:09:47 2019

@author: ngo
"""
import cv2
import numpy as np
import math

def BestAngle(img):
    AngleBest = 0
    MaxWidth =  -9999
    for AngleLoop in range(20):
        ListAmplitudeInside=[]
        Angle  = AngleLoop*math.pi/180
        for i in range(img.shape[1]):
            Amplitude = 0
            for ii in range(img.shape[0]):
                # print('Ver: {}, Hor: {}'.format(ii,int(min(0,i-ii*math.sin(Angle)))))
                if img[ii,int(max(0,i-ii*math.tan(Angle)))] == 0:
                    Amplitude += 1
            ListAmplitudeInside.append(Amplitude)
        CounterSpacePixel = 0
        SpaceWidthsInside = []
        for i, Amplitude in enumerate(ListAmplitudeInside):
            if Amplitude <= 0:
                CounterSpacePixel +=1
            else:
                if CounterSpacePixel >0:
                    SpaceWidthsInside.append(CounterSpacePixel)
                CounterSpacePixel=0
        # print('Line {}: Amplitudes {}'.format(IndexList,ListAmplitude))
        if len(SpaceWidthsInside)>0: 
            SpaceWidthsInside.pop(0)
        if len(SpaceWidthsInside)>0:
            # print('Best Angle: {} MaxSpace: {}'.format(AngleLoop,max                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            (SpaceWidthsInside)))
            if len(SpaceWidthsInside) > MaxWidth:
                MaxWidth = len(SpaceWidthsInside)
                AngleBest  = AngleLoop
    return AngleBest

def Dilate(img,ThresholdNumber,KernelWidth,KernelHeight):
    #Dilate
    AngleBest = BestAngle(img)            
    KernelAngel = AngleBest*math.pi/180
    ListWhitePoint = []
    _, thresh1 = cv2.threshold(img, 175, 255, cv2.THRESH_BINARY_INV)
    for i in range(thresh1.shape[0]):
        for ii in range(thresh1.shape[1]):
            if thresh1[i,ii] == 255:
                ListWhitePoint.append([i,ii])
                
    for Point1, Point2 in ListWhitePoint:
        for i in range(int(Point1-KernelHeight/2),int(Point1+KernelHeight/2)):    
            for ii in range(int(Point2-KernelWidth/2),int(Point2+KernelWidth/2)):
                 thresh1[max(0,i),int(max(0,ii-(i-Point1)*math.tan(KernelAngel)))] = 255
                 #print('i {} ii {}'.format(i,ii))
    for ii in range(thresh1.shape[1]):
        Count = 0
        for i in range(thresh1.shape[0]):
            Index1 = i
            Index2 = int(ii - i*math.tan(AngleBest*math.pi/180))
            if Index2 >= 0 and Index2 < thresh1.shape[1]:
                if thresh1[Index1, Index2] == 255:
                    Count += 1
        print('Count {} index1 {}'.format(Count,ii))
        if Count < ThresholdNumber:
            for i in range(thresh1.shape[0]):
                Index1 = i
                Index2 = int(ii - i*math.tan(AngleBest*math.pi/180))
                if Index2 >= 0 and Index2 < thresh1.shape[1]:
                    thresh1[Index1, Index2] = 0
    return thresh1

def SplitWords(img,ThresholdNumber,KernelWidth,KernelHeight):
    _, thresh1 = cv2.threshold(img, 175, 255, cv2.THRESH_BINARY)
    thresh1 = np.array(thresh1)
    thresh1 = Dilate(thresh1, ThresholdNumber, KernelWidth, KernelHeight)
    Contours, hierarchy = cv2.findContours(image= thresh1, mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_TC89_L1)
    BoundingContours = []
    for Contour in Contours:
        x,y,w,h = cv2.boundingRect(Contour)
        BoxContour = {'x':x,'y':y,'w':w,'h':h}
        BoundingContours.append(BoxContour)
    
    BoundingContours = sorted(BoundingContours,key= lambda X: X['x'])
    SplittedWords =[]
    Count = 0
    for Index, BoxContour in enumerate(BoundingContours):
        ImgWrite = img[BoxContour['y']:(BoxContour['y'] + BoxContour['h']),BoxContour['x']:(BoxContour['x']+BoxContour['w'])]
        ImgShow = thresh1[BoxContour['y']:(BoxContour['y'] + BoxContour['h']),BoxContour['x']:(BoxContour['x']+BoxContour['w'])]
        # cv2.imshow('',ImgShow)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        CounterWhite = 0
        for i in range(ImgShow.shape[0]):
            for ii in range(ImgShow.shape[1]):
                if ImgShow[i][ii] == 255:
                    CounterWhite +=1
        if CounterWhite >= 0.15 *ImgShow.shape[0] *ImgShow.shape[1] :
            # print(ImgShow)
            SplittedWords.append(ImgWrite)
            cv2.imwrite('./Result/Words{}.png'.format(Count),ImgWrite)
            Count += 1
    return SplittedWords

#th = 5 kw = 3 Test 1 Line.png
#th = 8 kw = 5 Test 2 Line0.png
ThresholdNumber = 7
KernelWidth = 3
KernelHeight = 4
imgReal = cv2.imread('./Line.png',0)
Words = SplitWords(imgReal,ThresholdNumber,KernelWidth,KernelHeight)
