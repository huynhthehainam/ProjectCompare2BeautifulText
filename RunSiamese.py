from Model import SiameseModel
import cv2
import numpy as np
import os

os.system('cls')
Image1 = cv2.imread('./Test1.png')
Image2 = cv2.imread('./Test2.png')
model  = SiameseModel()
model.LoadRawData('./Data')
model.Train('WeightProject.h5')

# model.LoadWeight('./WeightProject.h5')
# Pred = model.PredictOnePairImage(Image1,Image2)
# print(Pred)
