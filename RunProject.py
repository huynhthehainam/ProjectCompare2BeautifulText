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
        ListWordImage1 = self.SplitWordCombine(Image1,145,(4,20))
        # ListWordImage2 = self.SplitWordCombine(Image2,145,(3,20))
        # for i in range(len(ListWordImage1)):
        #     ListWordImage1[i] = cv2.cvtColor(ListWordImage1[i],cv2.COLOR_BGR2GRAY)
        #     ListWordImage1[i] = np.reshape(ListWordImage1[i],(124,124,1))/255
        #     # ListWordImage1[i] = cv2.threshold(ListWordImage1[i],175,255)
        # for i in range(len(ListWordImage2)):
        #     ListWordImage2[i] = cv2.cvtColor(ListWordImage2[i],cv2.COLOR_BGR2GRAY)
        #     ListWordImage2[i] = np.reshape(ListWordImage2[i],(124,124,1))/255
        #     # ListWordImage2[i] = cv2.threshold(ListWordImage2[i],175,255)
        # CounterSame = 0
        # IndexNext = 0
        # for Index1 in range(len(ListWordImage1)):
        #     IsSame = False
        #     for RangeIndex in range(-1,6):
        #         IndexNext = Index1
        #         Index2 = min(max(0, IndexNext + RangeIndex), len(ListWordImage2)-1)
        #         Pred = self.Model.PredictOnePairImage(ListWordImage1[Index1],ListWordImage2[Index2])
        #         if Pred[0][0] >0.2:
        #             IsSame = True
        #             IndexNext = Index2 +1
        #     if IsSame:
        #         CounterSame+=1
                

        #     else:
        #         for RangeIndex in range(-1,6):
        #             IndexNext = Index1
        #             Index2 = min(max(0, IndexNext + RangeIndex), len(ListWordImage2)-1)
        #             Pred = self.Model.PredictOnePairImage(ListWordImage1[Index1],ListWordImage2[Index2])
        #             ConcatImage = np.concatenate([ListWordImage1[Index1],ListWordImage2[Index2]], axis=1)
        #             cv2.imshow('{}'.format(Pred[0][0]),ConcatImage)
        #             cv2.waitKey(0)
        #             cv2.destroyAllWindows()
        # return CounterSame/len(ListWordImage1)*100
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
            if CounterWhite >= 0.15 *ImgShow.shape[0] *ImgShow.shape[1] :
                # print(ImgShow)
                SplittedWords.append(ImgWrite)
        return SplittedWords

    
    def SplitWordCombine (self, Image, ThresholdNumber, Kernel):
        files = glob.glob('./Result/*')
        for f in files:
            os.remove(f)
        SplittedLines = self.SplitLine(Image, ThresholdNumber,(15,3))
        ListResizedCroppedRealWord = []
        for IndexList in range(len(SplittedLines)):
            LineImg =  SplittedLines[IndexList]
            MinWord = -1
            SplittedWordsOut = []
            SplittedWordsOut = self.SplitWord(LineImg, ThresholdNumber,Kernel)
            for i, Img in enumerate(SplittedWordsOut):
                ResizedImage = np.zeros((max(Img.shape[0],Img.shape[1]),max(Img.shape[0],Img.shape[1]),3),np.uint8)
                ResizedImage[:] = (255,255,255)
                VerticalStartPoint =  int((ResizedImage.shape[0] - Img.shape[0])/2)
                HorizonalStartPoint =  int((ResizedImage.shape[1] - Img.shape[1])/2)
                for VerI in range(Img.shape[0]):
                    for HorI in range(Img.shape[1]):
                        ResizedImage[VerI+VerticalStartPoint,HorI + HorizonalStartPoint] = Img[VerI,HorI]

                ResizedImage = cv2.resize(ResizedImage, (124,124))
                ListResizedCroppedRealWord.append(ResizedImage)
        for i,Image in enumerate(ListResizedCroppedRealWord):
            cv2.imwrite('./Result/Word{}.png'.format(i),Image)
        print(len(ListResizedCroppedRealWord))
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