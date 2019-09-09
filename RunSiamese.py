from Model import SiameseModel
import cv2
import numpy as np
import os

os.system('cls')


model  = SiameseModel()
model.LoadRawData('./Data')
model.Train('WeightProject.h5')
# Pred = model.PredictOnePairImage(Image1,Image2)
