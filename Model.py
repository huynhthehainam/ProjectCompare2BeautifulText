from keras.models import Sequential
import time
from keras.optimizers import Adam
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate, Dropout
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import *
from keras.engine.topology import Layer
from keras import backend as K
from keras.regularizers import l2
K.set_image_data_format('channels_last')
import cv2
import os
from skimage import io
import numpy as np
import tensorflow as tf
import numpy.random as rng
from sklearn.utils import shuffle
import glob
import random



class SiameseModel:
    NumberTotalData = 200
    BatchSize = 32
    Epochs = 200
    def initialize_weights(self,shape, name=None):
        return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)
    def initialize_bias(self,shape, name=None):
        return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)
    def get_siamese_model(self,input_shape):
        left_input = Input(input_shape)
        right_input = Input(input_shape)
        model = Sequential()
        model.add(Conv2D(64, (10,10), activation='relu', input_shape=input_shape,
                    kernel_initializer=self.initialize_weights, kernel_regularizer=l2(2e-4)))
        model.add(MaxPooling2D())
        model.add(Conv2D(128, (7,7), activation='relu',
                        kernel_initializer=self.initialize_weights,
                        bias_initializer=self.initialize_bias, kernel_regularizer=l2(2e-4)))
        model.add(MaxPooling2D())
        model.add(Conv2D(128, (4,4), activation='relu', kernel_initializer=self.initialize_weights,
                        bias_initializer=self.initialize_bias, kernel_regularizer=l2(2e-4)))
        model.add(MaxPooling2D())
        model.add(Conv2D(256, (4,4), activation='relu', kernel_initializer=self.initialize_weights,
                        bias_initializer=self.initialize_bias, kernel_regularizer=l2(2e-4)))
        model.add(Flatten())
        model.add(Dense(4096, activation='sigmoid',
                    kernel_regularizer=l2(1e-3),
                    kernel_initializer=self.initialize_weights,bias_initializer=self.initialize_bias))
        encoded_l = model(left_input)
        encoded_r = model(right_input)
        L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
        L1_distance = L1_layer([encoded_l, encoded_r])
        prediction = Dense(1,activation='sigmoid',bias_initializer=self.initialize_bias)(L1_distance)
        siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
        return siamese_net
    def __init__(self):
        self.Model = self.get_siamese_model((124,124,3))
        self.Optimizer = Adam(lr = 0.00006)
        self.Model.compile(loss="binary_crossentropy",optimizer=self.Optimizer)
    def LoadRawData(self,DataPath):
        DataDir = os.path.join(DataPath)
        Folders = os.listdir(DataDir)
        self.Data = []
        for Folder in Folders:
            FolderData = []
            FolderDir = os.path.join(DataDir,Folder)
            FileNames = os.listdir(os.path.join('Data',Folder))
            for FileName in FileNames:
                FolderData.append(cv2.imread(os.path.join(FolderDir,FileName)))
            self.Data.append(FolderData)
        self.ProcessRawData()
        print('Finish Load Data')

        return True
    def ProcessRawData(self):
        print('Start creating batch')
        self.InputByLocation = []
        self.Labels = []
        for i in range(self.NumberTotalData):
            RandFolders = np.array(random.sample(list(range(len(self.Data))),k = 2))
            Folder1 = RandFolders[0]
            Folder2 = RandFolders[1]
            RandImages = np.array(random.sample(list(range(10)),k = 2))
            RandImage1 = RandImages[0]
            RandImage2 = RandImages[1]
            Location  = [[Folder1,RandImage1],[Folder1,RandImage2]]
            if Location not in self.InputByLocation:
                self.InputByLocation.append(Location)
                self.Labels.append(1)
            RandImages = np.array(random.sample(list(range(10)),k = 2))
            RandImage1 = RandImages[0]
            RandImage2 = RandImages[1]
            Location  = [[Folder1,RandImage1],[Folder2,RandImage2]]
            if Location not in self.InputByLocation:
                self.InputByLocation.append(Location)
                if Folder1 == Folder2:
                    self.Labels.append(1)
                else:
                    self.Labels.append(0)
        print('Finish Process Raw Data')
        return True
    def GetBatch(self):
        Labels = []
        InBatchLabel = []
        Pair = []
        BatchLocation0=[]
        BatchLocation1=[]
        for i in range(len(self.InputByLocation)):
            if i % self.BatchSize !=0:
                BatchLocation0.append(self.InputByLocation[i][0])
                BatchLocation1.append(self.InputByLocation[i][1])
                InBatchLabel.append(self.Labels[i])
            else:
                if len(InBatchLabel)!=0:
                    Pair.append([BatchLocation0,BatchLocation1])
                    BatchLocation0 = []
                    BatchLocation1 =[]
                    Labels.append(InBatchLabel)
                    InBatchLabel = []
                BatchLocation0.append(self.InputByLocation[i][0])
                BatchLocation1.append(self.InputByLocation[i][1])
                InBatchLabel.append(self.Labels[i])
        return Pair, Labels
    def GetImageFromLocation(self, Location):
        return self.Data[Location[0]][Location[1]]
    
    def ConvertPairLocationToPairImage(self, Pair):
        ImageList0=np.array([self.GetImageFromLocation(Location) for Location in Pair[0]])
        ImageList1 = np.array([self.GetImageFromLocation(Location) for Location in Pair[1]])
        return [ImageList0,ImageList1]


    
    def Train(self, SaveModelPath = None):
        print('Start training')
        Pairs, Labels = self.GetBatch()
        #print(len(Labels[0]))
        for  i  in range(self.Epochs):
            for ii in range(len(Pairs)):
                X = self.ConvertPairLocationToPairImage(Pairs[ii])
                Y = np.array(Labels[ii])
                loss = self.Model.train_on_batch(X,Y)
            print('Epochs {} Loss: {}'.format(i,loss))
            self.TestOneShot()
            if SaveModelPath:
                self.Model.save_weights(SaveModelPath)
        print('Train finished')
        return True

    def LoadWeight(self, WeightPath):
        print('Load weight from: {}'.format(WeightPath))
        self.Model.load_weights(WeightPath)
        return True
    
    def TestOneShot(self):
        RandFolders = np.array(random.sample(list(range(len(self.Data))),k = 2))
        Folder1 = RandFolders[0]
        Folder2 = RandFolders[1]
        RandImages = np.array(random.sample(list(range(10)),k = 2))
        RandImage1 = RandImages[0]
        RandImage2 = RandImages[1]
        Image1 = self.Data[Folder1][RandImage1]
        Image2 = self.Data[Folder1][RandImage2]
        Label = 1
        print('Label: {}, Predict: {}'.format(Label,self.PredictOnePairImage(Image1,Image2)))
        Image2 = self.Data[Folder2][RandImage2]
        if Folder1 == Folder2:
            Label = 1
        else:
            Label = 0
        print('Label: {}, Predict: {}'.format(Label,self.PredictOnePairImage(Image1,Image2)))
        return True
          

    def PredictOnePairImage(self, Image1, Image2):
        Image1 = np.array([Image1])
        Image2 = np.array([Image2])
        return self.Model.predict_on_batch([Image1,Image2])

        
        
        


        

