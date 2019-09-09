from Model import SiameseModel
import cv2
import numpy as np

Image1 =  cv2.imread('TestModel1.png')
Image2 = cv2.imread('TestModel2.png')

model  = SiameseModel()
Pred = model.PredictOnePairImage(Image1,Image2)
print(Pred)