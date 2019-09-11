import cv2
import math
import numpy as np
import glob
import os
from Model import SiameseModel

class CompareTwoTextImage:
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
    def CompareTwoImage(self, Image1Path, Image2Path):
        Image1 = cv2.imread(Image1Path)

        Image2 = cv2.imread(Image2Path)
        ListWordImage1 = self.SplitWord(Image1)
        ListWordImage2 = self.SplitWord(Image2)
        CounterSame = 0
        for Index1 in range(len(ListWordImage1)):
            IsSame = False
            for RangeIndex in range(-2,3):
                Index2 = min(max(0, Index1 + RangeIndex), len(ListWordImage2)-1)
                Pred = self.Model.PredictOnePairImage(ListWordImage1[Index1],ListWordImage2[Index2])
                ConcatImage = np.concatenate([ListWordImage1[Index1],ListWordImage2[Index2]], axis=1)
                cv2.imshow('{}'.format(Pred[0][0]),ConcatImage)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                if Pred[0][0] >0.7:
                    IsSame = True
            if IsSame:
                CounterSame+=1
        return CounterSame/len(ListWordImage1)*100
    def SplitWord(self, Image):
        files = glob.glob('./Result/*')
        for f in files:
            os.remove(f)
        ThreshHold1 = 0
        ThreshHold2 = 0
        Hihi=[]
        RealImage = Image
        grayImage = cv2.cvtColor(RealImage, cv2.COLOR_BGR2GRAY)
        _, thresh1 = cv2.threshold(grayImage, 175, 255, cv2.THRESH_BINARY)
        ListAmplitude = []
        for i in range(thresh1.shape[0]):
            Amplitude = 0
            for ii in range(thresh1.shape[1]):
                if thresh1[i, ii] == 0:
                    Amplitude += 1
            ListAmplitude.append(Amplitude)
        ListAmplitude = self.ThesholdList(ListAmplitude,ThreshHold1)
        CounterSpacePixel = 0
        SpaceHeights = []
        for i, Amplitude in enumerate(ListAmplitude):
            if Amplitude <= 0:
                CounterSpacePixel +=1
            else:
                if CounterSpacePixel >0:
                    SpaceHeights.append(CounterSpacePixel)
                CounterSpacePixel=0
        # print(SpaceHeights)

        CurrentAmplitude = 0
        ListCroppedLine = []
        ListCroppedRealLine = []
        StartPoint = 0
        EndPoint = 0
        SpaceHeight = 5
        if len(SpaceHeights)>0:
            SpaceHeights.pop(0)
        if len (SpaceHeights)>0:
            SpaceHeight = math.ceil(self.MiddleOfList(SpaceHeights)*0.2)
        IsDetectEndPoint = True
        for i, Amplitude in enumerate(ListAmplitude):
            if Amplitude > 0 and CurrentAmplitude <= 0 and IsDetectEndPoint:
                StartPoint = i
                IsDetectEndPoint = False
            if Amplitude <= 0 and CurrentAmplitude > 0:
                IsCoupleOfPixelABlack = False
                for ii in range(SpaceHeight):
                    if i+ii<len(ListAmplitude) and ListAmplitude[i+ii]>0:
                        IsCoupleOfPixelABlack = True
                if not IsCoupleOfPixelABlack:
                    IsDetectEndPoint = True
                    EndPoint = i
                    ListCroppedLine.append(thresh1[max(StartPoint,0):min(EndPoint,thresh1.shape[0]), :])
                    ListCroppedRealLine.append(RealImage[max(StartPoint,0):min(EndPoint,thresh1.shape[0]), :])
            CurrentAmplitude = Amplitude
        
        ListResizedCroppedRealWord = []
        for IndexList in range(len(ListCroppedLine)):
            img =  ListCroppedLine[IndexList]
            imgReal = ListCroppedRealLine[IndexList]
            # cv2.imwrite('./Result/Line{}.png'.format(IndexList),img)
            ListAmplitude = []
            ListCroppedWord = []
            ListCroppedRealWord =[]
            MaxWidth =  -9999
            SpaceWidths = []
            AngleBest = 0
            for AngleLoop in range(20):
                ListAmplitudeInside=[]
                Angle  = AngleLoop*math.pi/180
                for i in range(img.shape[1]):
                    Amplitude = 0
                    for ii in range(img.shape[0]):
                        # print('Ver: {}, Hor: {}'.format(ii,int(min(0,i-ii*math.sin(Angle)))))
                        if img[ii,int(max(0,i-ii*math.sin(Angle)))] == 0:
                            Amplitude += 1
                    ListAmplitudeInside.append(Amplitude)
                StartPoint = 0
                EndPoint = 0
                
                BeforeAmplitude = 0
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
                        SpaceWidths = SpaceWidthsInside
                        ListAmplitude = ListAmplitudeInside
                        AngleBest  = AngleLoop
            WordWidths = []
            CounterWordPixel = 0
            # print('Amplitudes: {}'.format(ListAmplitude))
            for i, Amplitude in enumerate(ListAmplitude):
                if Amplitude > 0:
                    CounterWordPixel +=1
                else:
                    if CounterWordPixel >0:
                        WordWidths.append(CounterWordPixel)
                    CounterWordPixel=0
            SpaceWidth = 3
            if len(WordWidths)>0:
                # print('Line: {} Best Angle: {} WordWidth: {}'.format(IndexList,AngleBest, WordWidths))
                # print('MaxSpaceWidth: {} MaxWordWidth: {}'.format(max(SpaceWidths),max(WordWidths)))
                # print('Height: {}'.format(img.shape[0]))
                Height =  img.shape[0]
            
                SpaceWidth  = math.ceil(self.MiddleOfList(WordWidths)*0.3)
            # if max(WordWidths)<Height:
            #     SpaceWidth = self.MiddleOfList(SpaceWidths)
            # SpaceWidth =  min(SpaceWidths)
            # print('SpaceWidth: {}'.format(SpaceWidth))
            # ShowAmplitudeBar(ListAmplitude)
            IsDetectEndPoint = True
            CurrentAmplitude = 0
            # print(AngleBest)
            AngleBest =  AngleBest*math.pi/180
            ListAmplitude = self.ThesholdList(ListAmplitude,ThreshHold2)
            for i, Amplitude in enumerate(ListAmplitude):
                if Amplitude > 0 and CurrentAmplitude <= 0 and IsDetectEndPoint:
                    for NextPixel in range(1,SpaceWidth):
                        if ListAmplitude[i - NextPixel]>0 or i - NextPixel<0:
                            break
                    StartPoint = i - int(NextPixel/2)
                    IsDetectEndPoint = False
                if Amplitude <= 0 and CurrentAmplitude > 0:
                    IsCoupleOfPixelABlack = False
                    for ii in range(SpaceWidth):
                        if i+ii<len(ListAmplitude) and ListAmplitude[i+ii]>0:
                            IsCoupleOfPixelABlack = True
                    if not IsCoupleOfPixelABlack:
                        IsDetectEndPoint = True
                        for NextPixel in range(SpaceWidth):
                            if ListAmplitude[i + NextPixel]>0 or i+NextPixel>=len(ListAmplitude):
                                break
                        EndPoint = i + int(NextPixel/2)
                        # print('StartPoint: {}, EndPoint: {}'.format(StartPoint,EndPoint))
                        ListCroppedWord.append(img[:,max(StartPoint-int(img.shape[0]*math.sin(AngleBest)),0):min(EndPoint,img.shape[1])])
                        ListCroppedRealWord.append(imgReal[:,max(StartPoint-int(img.shape[0]*math.sin(AngleBest)),0):min(EndPoint,img.shape[1])])
                CurrentAmplitude = Amplitude
           
            for i, Img in enumerate(ListCroppedRealWord):
                Hihi.append(Img.shape[1]/Img.shape[0])
                # print('-----------{}-----------'.format(Img.shape[1]/Img.shape[0]))
                # print('{} {}'.format(Img.shape[1],Img.shape[0]))
                ResizedImage = np.zeros((max(Img.shape[0],Img.shape[1]),max(Img.shape[0],Img.shape[1]),3),np.uint8)
                ResizedImage[:] = (255,255,255)
                VerticalStartPoint =  int((ResizedImage.shape[0] - Img.shape[0])/2)
                HorizonalStartPoint =  int((ResizedImage.shape[1] - Img.shape[1])/2)
                for VerI in range(Img.shape[0]):
                    for HorI in range(Img.shape[1]):
                        ResizedImage[VerI+VerticalStartPoint,HorI + HorizonalStartPoint] = Img[VerI,HorI]
                ResizedImage = cv2.resize(ResizedImage, (124,124))
                ListResizedCroppedRealWord.append(ResizedImage)
                # cv2.imwrite('./Result/Line{}Word{}.png'.format(IndexList,i),ResizedImage)
        return ListResizedCroppedRealWord
ModelSiamese = ''
ModelSiamese = SiameseModel()
ModelSiamese.LoadWeight('WeightProject.h5')
CompareCore = CompareTwoTextImage(ModelSiamese)
print(CompareCore.CompareTwoImage('TestProject1.PNG','TestProject2.PNG'))