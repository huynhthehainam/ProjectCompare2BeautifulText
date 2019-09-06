# from keras.models import Sequential
# import time
# from keras.optimizers import Adam
# from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
# from keras.models import Model
# from keras.layers.normalization import BatchNormalization
# from keras.layers.pooling import MaxPooling2D, AveragePooling2D
# from keras.layers.merge import Concatenate
# from keras.layers.core import Lambda, Flatten, Dense
# from keras.initializers import glorot_uniform
# from sklearn.preprocessing import LabelBinarizer
# from keras.optimizers import *
# from keras.engine.topology import Layer
# from keras import backend as K
# from keras.regularizers import l2
# K.set_image_data_format('channels_last')
import cv2
import os
from skimage import io
import numpy as np
import tensorflow as tf
import numpy.random as rng
from sklearn.utils import shuffle
import glob
from random import random



class SiameseModel:
    NumberTotalData = 200
    BatchSize = 32
    Epochs = 200
    def initialize_weights(self,shape, name=None):
        return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)
    def initialize_bias(self,shape, name=None):
        return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)
    # def get_siamese_model(self,input_shape):
    #     """
    #     Model architecture based on the one provided in: http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
    #     """
    #     left_input = Input(input_shape)
    #     right_input = Input(input_shape)
    #     model = Sequential()
    #     model.add(Conv2D(64, (10,10), activation='relu', input_shape=input_shape,
    #                 kernel_initializer=self.initialize_weights, kernel_regularizer=l2(2e-4)))
    #     model.add(MaxPooling2D())
    #     model.add(Conv2D(128, (7,7), activation='relu',
    #                     kernel_initializer=self.initialize_weights,
    #                     bias_initializer=self.initialize_bias, kernel_regularizer=l2(2e-4)))
    #     model.add(MaxPooling2D())
    #     model.add(Conv2D(128, (4,4), activation='relu', kernel_initializer=self.initialize_weights,
    #                     bias_initializer=self.initialize_bias, kernel_regularizer=l2(2e-4)))
    #     model.add(MaxPooling2D())
    #     model.add(Conv2D(256, (4,4), activation='relu', kernel_initializer=self.initialize_weights,
    #                     bias_initializer=self.initialize_bias, kernel_regularizer=l2(2e-4)))
    #     model.add(Flatten())
    #     model.add(Dense(4096, activation='sigmoid',
    #                 kernel_regularizer=l2(1e-3),
    #                 kernel_initializer=self.initialize_weights,bias_initializer=self.initialize_bias))
    #     encoded_l = model(left_input)
    #     encoded_r = model(right_input)
    #     L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    #     L1_distance = L1_layer([encoded_l, encoded_r])
    #     prediction = Dense(1,activation='sigmoid',bias_initializer=self.initialize_bias)(L1_distance)
    #     siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    #     return siamese_net
    # def __init__(self):
    #     self.model = self.get_siamese_model((124,124,3))
    #     self.optimizer = Adam(lr = 0.00006)
    #     self.model.compile(loss="binary_crossentropy",optimizer=self.optimizer)
    def LoadRawData(self):
        DataDir = os.path.join('Data')
        Folders = os.listdir(DataDir)
        self.Data = []
        for Folder in Folders:
            FolderData = []
            FolderDir = os.path.join(DataDir,Folder)
            FileNames = os.listdir(os.path.join('Data',Folder))
            for FileName in FileNames:
                FolderData.append(cv2.imread(os.path.join(FolderDir,FileName)))
            self.Data.append(FolderData)
        print('Finish Load Data')

        return True
    def ProcessRawData(self):
        self.InputByLocation = []
        self.Labels = []
        for i in range(self.NumberTotalData):
            RandFolders = np.random.randint(len(self.Data),size = 2)
            Folder1 = RandFolders[0]
            Folder2 = Folder1
            RandImages =  np.random.randint(9,size = 2)
            RandImage1 = RandImages[0]
            RandImage2 = RandImages[1]
            Location  = [Folder1,RandImage1,Folder2,RandImage2]
            if Location not in self.InputByLocation:
                self.InputByLocation.append(Location)
                if Folder1 == Folder2:
                    self.Labels.append(1)
                else:
                    self.Labels.append(0)
        for i in range(self.NumberTotalData):
            RandFolders = np.random.randint(len(self.Data),size = 2)
            Folder1 = RandFolders[0]
            Folder2 = RandFolders[1]
            RandImages =  np.random.randint(9,size = 2)
            RandImage1 = RandImages[0]
            RandImage2 = RandImages[1]
            Location  = [Folder1,RandImage1,Folder2,RandImage2]
            if Location not in self.InputByLocation:
                self.InputByLocation.append(Location)
                if Folder1 == Folder2:
                    self.Labels.append(1)
                else:
                    self.Labels.append(0)
        print('Finish Process Raw Data')
        return True
    def GetBatch(self):
        BatchData =  []
        InBatchData = []
        Labels = []
        InBatchLabel = []
        for i in range(len(self.InputByLocation)):
            if i % self.BatchSize !=0:
                InBatchData.append(self.InputByLocation[i])
                InBatchLabel.append(self.Labels[i])
            else:
                if len(InBatchData)!=0:
                    BatchData.append(InBatchData)
                    Labels.append(InBatchLabel)
                    InBatch = []
        return BatchData, Labels
    def GetImageFromBatch(self, BatchData):
        for i in range(len(BatchData)):
            Image1 = self.Data[BatchData[i][0]][BatchData[i][1]]
            Image2 = self.Data[BatchData[i][2]][BatchData[i][3]]
            cv2.imshow('Image1',Image1)
            cv2.imshow('Image2',Image2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    def Train(self, SaveModelPath = None):
        BatchData, Labels = self.GetBatch()
        # self.GetImageFromBatch(BatchData[0])
        print(Labels[0])
        # for i in range(self.Epochs):
        #     for ii in range(len(BatchData)):

        
        
        


        

