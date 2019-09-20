import cv2
import math
import numpy as np
import glob
import os
import copy
from Model import SiameseModel

class CompareTwoTextImage:
    ThresholdNumber = 145
    def __init__(self, ModelSiamese):
        self.Model = ModelSiamese
    def AverageOfList(self,ListNumber):
        return int(sum(ListNumber)/len(ListNumber))
    def MiddleOfList(self,ListNumber):
        ListNumber = sorted(ListNumber)
        return ListNumber[int(len(ListNumber)/2)]
    def ThesholdList(self,ListNumber, ThreshNumber):
        for i in range(len(ListNumber)):
            if ListNumber[i] < ThreshNumber:
                ListNumber[i] = 0
        return ListNumber
    SpreadWidth = 2
    def CompareTwoImage(self, Image1Path, Image2Path):
        Image1 = cv2.imread(Image1Path)
        Image2 = cv2.imread(Image2Path)
        ListWordImage1 = self.SplitWordCombine(Image1,145,(4,20),0)
        ListWordImage2 = self.SplitWordCombine(Image2,145,(4,20),0)
        for i in range(len(ListWordImage1)):
            ListWordImage1[i] = cv2.cvtColor(ListWordImage1[i],cv2.COLOR_BGR2GRAY)
            ListWordImage1[i] = np.reshape(ListWordImage1[i],(124,124,1))
            # ListWordImage1[i] = cv2.threshold(ListWordImage1[i],175,255)
        for i in range(len(ListWordImage2)):
            ListWordImage2[i] = cv2.cvtColor(ListWordImage2[i],cv2.COLOR_BGR2GRAY)
            ListWordImage2[i] = np.reshape(ListWordImage2[i],(124,124,1))
            # ListWordImage2[i] = cv2.threshold(ListWordImage2[i],175,255)
        print(len(ListWordImage1))
        print(len(ListWordImage2))
        CounterSame = 0
        IndexNext = 0
        for Index1 in range(len(ListWordImage1)):
            IsSame = False
            for RangeIndex in range(-1,6):
                IndexNext = Index1
                Index2 = min(max(0, IndexNext + RangeIndex), len(ListWordImage2)-1)
                Pred = self.Model.PredictOnePairImage(ListWordImage1[Index1],ListWordImage2[Index2])
                if Pred[0][0] >0.5:
                    IsSame = True
                    IndexNext = Index2 +1
                    ConcatImage = np.concatenate([ListWordImage1[Index1],ListWordImage2[Index2]], axis=1)
                    cv2.imshow('{}'.format(Pred[0][0]),ConcatImage)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            if IsSame:
                CounterSame+=1
                

            else:
                for RangeIndex in range(-1,6):
                    IndexNext = Index1
                    Index2 = min(max(0, IndexNext + RangeIndex), len(ListWordImage2)-1)
                    Pred = self.Model.PredictOnePairImage(ListWordImage1[Index1],ListWordImage2[Index2])
                    ConcatImage = np.concatenate([ListWordImage1[Index1],ListWordImage2[Index2]], axis=1)
                    cv2.imshow('{}'.format(Pred[0][0]),ConcatImage)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
        return CounterSame/len(ListWordImage1)*100
        return True
    def SplitLine(self,Img, ThresholdNumber,Kernel):
        T = ThresholdNumber
        Gray  = cv2.cvtColor(src = Img,code= cv2.COLOR_BGR2GRAY)
        (T, Binary) = cv2.threshold(src = Gray,thresh=T,maxval=255,type=cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
        RectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, Kernel)
        Dilation = cv2.dilate(Binary, RectKernel, iterations =1)
        Contours, hierarchy = cv2.findContours(image= Dilation, mode=cv2.RETR_TREE,method=cv2.CHAIN_APPROX_SIMPLE)
        BoundingContours = []
        for Contour in Contours:
            x,y,w,h = cv2.boundingRect(Contour)
            BoxContour = {'x':x,'y':y,'w':w,'h':h}
            BoundingContours.append(BoxContour)

        BoundingContours = sorted(BoundingContours,key= lambda X: X['y'])
        SplittedLines =[]
        for Index, BoxContour in enumerate(BoundingContours):
            ImgWrite = Img[BoxContour['y']:(BoxContour['y'] + BoxContour['h']),BoxContour['x']:(BoxContour['x']+BoxContour['w'])]
            ImgShow = Binary[BoxContour['y']:(BoxContour['y'] + BoxContour['h']),BoxContour['x']:(BoxContour['x']+BoxContour['w'])]
            CounterWhite = 0
            for i in range(ImgShow.shape[0]):
                for ii in range(ImgShow.shape[1]):
                    if ImgShow[i][ii] == 255:
                        CounterWhite +=1
            if CounterWhite >= 0.15 *ImgShow.shape[0] *ImgShow.shape[1] :
                SplittedLines.append(ImgWrite)
        return SplittedLines
    def SplitWord(self,Img, ThresholdNumber, Kernel):
        T = ThresholdNumber
        Gray  = cv2.cvtColor(src = Img,code= cv2.COLOR_BGR2GRAY)
        (T, Binary) = cv2.threshold(src = Gray,thresh=T,maxval=255,type=cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
        RectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, Kernel)
        Dilation = cv2.dilate(Binary, RectKernel, iterations =1)

        Contours, hierarchy = cv2.findContours(image= Dilation, mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_TC89_L1)
        BoundingContours = []
        for Contour in Contours:
            x,y,w,h = cv2.boundingRect(Contour)
            BoxContour = {'x':x,'y':y,'w':w,'h':h}
            BoundingContours.append(BoxContour)

        BoundingContours = sorted(BoundingContours,key= lambda X: X['x'])
        SplittedWords =[]
        for Index, BoxContour in enumerate(BoundingContours):
            ImgWrite = Img[BoxContour['y']:(BoxContour['y'] + BoxContour['h']),BoxContour['x']:(BoxContour['x']+BoxContour['w'])]
            ImgShow = Binary[BoxContour['y']:(BoxContour['y'] + BoxContour['h']),BoxContour['x']:(BoxContour['x']+BoxContour['w'])]
            # cv2.imshow('',ImgShow)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            CounterWhite = 0
            for i in range(ImgShow.shape[0]):
                for ii in range(ImgShow.shape[1]):
                    if ImgShow[i][ii] == 255:
                        CounterWhite +=1
            if CounterWhite >= 0.15 *ImgShow.shape[0] *ImgShow.shape[1] and ImgShow.shape[1]*3 < ImgShow.shape[0]:
                # print(ImgShow)
                SplittedWords.append(ImgWrite)
        return SplittedWords
    def BestAngle(self,Img):
        AngleBest = 0
        MaxWidth =  -9999
        # cv2.imshow('',Img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        for AngleLoop in range(20):
            ListAmplitudeInside=[]
            Angle  = AngleLoop*math.pi/180
            for ii in range(Img.shape[1]):
                Amplitude = 0
                for i in range(Img.shape[0]):
                    Index1 = i
                    Index2 = int(max(0,ii-i*math.tan(Angle)))
                    if Index2 >=0 and Index2<Img.shape[1]:
                        if Img[i,int(max(0,ii-i*math.tan(Angle)))] == 0:
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
            if len(SpaceWidthsInside)>0: 
                SpaceWidthsInside.pop(0)
            if len(SpaceWidthsInside)>0:                                                                                                                                                                                                                                                                                                                                                                
                if len(SpaceWidthsInside) > MaxWidth +3:
                    # print(AngleLoop)
                    # print(ListAmplitudeInside)
                    MaxWidth = len(SpaceWidthsInside)
                    AngleBest  = AngleLoop
        
        return AngleBest
    
    def SplitWordCombine (self, Image, ThresholdNumber, Kernel, AdjustKernel):
        files = glob.glob('./Result/*')
        for f in files:
            os.remove(f)
        SplittedLines = self.SplitLine(Image, ThresholdNumber,(15,3))
        ListResizedCroppedRealWord = []
        for IndexList in range(len(SplittedLines)):
            OriginalImage = SplittedLines[IndexList]
            Gray = cv2.cvtColor(OriginalImage,cv2.COLOR_BGR2GRAY)
            # BestAngle = self.BestAngle(Gray)
            # print('Line: {} Angle: {}'.format(IndexList,BestAngle))
            KernelAngel = 0*math.pi/180
            KernelWidth = 4
            KernelHeight = 5
            ListWhitePoint = []
            _, Thresh = cv2.threshold(Gray, 175, 255, cv2.THRESH_BINARY_INV)
            for i in range(Thresh.shape[0]):
                for ii in range(Thresh.shape[1]):
                    if Thresh[i,ii] == 255:
                        ListWhitePoint.append([i,ii])
            cv2.imwrite('./Result/Line{}B.png'.format(IndexList),Thresh)   
            ThreshCopy = copy.deepcopy(Thresh)  
            for Point1, Point2 in ListWhitePoint:
                Point1Copy = Point1
                Point2Copy = Point2
                CounterMinusLeft = 0
                CounterLeft = 0
                CounterMinusRight = 0
                CounterRight = 0
                while True:
                    CounterMinusLeft +=1
                    if ThreshCopy[Point1,Point2 - CounterMinusLeft] == 255 or (Point2-CounterMinusLeft)<0:
                        break
                    CounterLeft+=1
                while True:
                    CounterMinusRight +=1
                    if ThreshCopy[Point1,Point2+ CounterMinusRight] == 255 or (Point2+CounterMinusRight)>=ThreshCopy.shape[1]-1:
                        break
                    CounterRight+=1
                # if Point1 == 5 and Point2 == 590:
                #     print('Line: {} Point: [{},{}] CounterLeft: {}, CounterRight: {}'.format(IndexList,Point1,Point2, CounterLeft, CounterRight))
                for i in range(int(Point1-KernelHeight/2),int(Point1+KernelHeight/2)):
                    if CounterLeft > CounterRight:
                        for ii in range(int(Point2-1),int(Point2+KernelWidth-1)):
                            # print('{}  {}'.format(i,ii))
                            Thresh[max(0,i),int(max(0,ii-(i-Point1)*math.tan(KernelAngel)))] = 255
                    elif CounterLeft == CounterRight:
                        pass
                        # for ii in range(int(Point2-KernelWidth/2),int(Point2+KernelWidth/2)):
                        #     Thresh[max(0,i),int(max(0,ii-(i-Point1)*math.tan(KernelAngel)))] = 255
                    else:
                        for ii in range(int(Point2-KernelWidth+1+AdjustKernel),int(Point2+1)):
                            Thresh[max(0,i),int(max(0,ii-(i-Point1)*math.tan(KernelAngel)))] = 255
            cv2.imwrite('./Result/Line{}.png'.format(IndexList),Thresh)
            # for ii in range(Thresh.shape[1]):
            #     Count = 0
            #     for i in range(Thresh.shape[0]):
            #         Index1 = i
            #         Index2 = int(ii - i*math.tan(BestAngle*math.pi/180))
            #         if Index2 >= 0 and Index2 < Thresh.shape[1]:
            #             if Thresh[Index1, Index2] == 255:
            #                 Count += 1
            #     if Count < ThresholdNumber:
            #         for i in range(Thresh.shape[0]):
            #             Index1 = i
            #             Index2 = int(ii - i*math.tan(BestAngle*math.pi/180))
            #             if Index2 >= 0 and Index2 < Thresh.shape[1]:
            #                 Thresh[Index1, Index2] = 0
            Contours, hierarchy = cv2.findContours(image= Thresh, mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_TC89_L1)
            BoundingContours = []
            for Contour in Contours:
                x,y,w,h = cv2.boundingRect(Contour)
                BoxContour = {'x':x,'y':y,'w':w,'h':h}
                BoundingContours.append(BoxContour)

            BoundingContours = sorted(BoundingContours,key= lambda X: X['x'])
            SplittedWords =[]
            for Index, BoxContour in enumerate(BoundingContours):
                ImgWrite = OriginalImage[BoxContour['y']:(BoxContour['y'] + BoxContour['h']),BoxContour['x']:(BoxContour['x']+BoxContour['w'])]
                ImgShow = Thresh[BoxContour['y']:(BoxContour['y'] + BoxContour['h']),BoxContour['x']:(BoxContour['x']+BoxContour['w'])]
                # cv2.imshow('',ImgShow)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                CounterWhite = 0
                for i in range(ImgShow.shape[0]):
                    for ii in range(ImgShow.shape[1]):
                        if ImgShow[i][ii] == 255:
                            CounterWhite +=1
                if CounterWhite >= 0.18 *ImgShow.shape[0] *ImgShow.shape[1] :
                    # print(ImgShow)
                    SplittedWords.append(ImgWrite)
            for i, ImgWrite in enumerate(SplittedWords):
                ResizedImage = np.zeros((max(ImgWrite.shape[0],ImgWrite.shape[1]),max(ImgWrite.shape[0],ImgWrite.shape[1]),3),np.uint8)
                ResizedImage[:] = (255,255,255)
                VerticalStartPoint =  int((ResizedImage.shape[0] - ImgWrite.shape[0])/2)
                HorizonalStartPoint =  int((ResizedImage.shape[1] - ImgWrite.shape[1])/2)
                for VerI in range(ImgWrite.shape[0]):
                    for HorI in range(ImgWrite.shape[1]):
                        ResizedImage[VerI+VerticalStartPoint,HorI + HorizonalStartPoint] = ImgWrite[VerI,HorI]

                ResizedImage = cv2.resize(ResizedImage, (124,124))
                ListResizedCroppedRealWord.append(ResizedImage)
                cv2.imwrite('./Result/Line{}Word{}.png'.format(IndexList,i),ResizedImage)
        return ListResizedCroppedRealWord

    def SpreadList(self,ListNumber,SpreadWidth):
        ListAdd =[]
        for i in range(len(ListNumber)):
            if ListNumber[i]>0:
                ListAdd.append(i)
        for Index in ListAdd:
            for ii in range(SpreadWidth):
                IndexFill = Index - ii
                IndexFill = min(max(IndexFill,0), len(ListNumber)-1)
                ListNumber[IndexFill] += 5
        return ListNumber
ModelSiamese = ''
ModelSiamese = SiameseModel()
ModelSiamese.LoadWeight('WeightProject.h5')
CompareCore = CompareTwoTextImage(ModelSiamese)
print(CompareCore.CompareTwoImage('TestProject1.png','TestProject2.PNG'))