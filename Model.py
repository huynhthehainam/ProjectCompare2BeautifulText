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
    BatchSize = 32
    Iteration = 200000
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
        model.add(Dense(8192, activation='sigmoid',
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
        self.Model = self.get_siamese_model((124,124,1))
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
                Image = np.array(cv2.imread(os.path.join(FolderDir,FileName),0), dtype = 'float32')/255
                ReshapedImage = np.reshape(Image,(124, 124, 1))
                FolderData.append(ReshapedImage)
            self.Data.append(FolderData)
        #self.ProcessRawData()
        print('Finish Load Data')

        return True
    def GetBatch(self, Iteration):
        #print('Start creating batch')
        InputByLocation = []
        Labels = []
        ImageLeft = []
        ImageRight = []
        Folder1 = Iteration%len(self.Data)
        IndexOfFolders = list(range(len(self.Data)))
        IndexOfFolders.remove(Folder1)
        _, Folder2 = np.array(random.sample(IndexOfFolders,k = 2))
        for ii in range(24):
            RandImages = np.array(random.sample(list(range(10)),k = 2))
            RandImage1 = RandImages[0]
            RandImage2 = RandImages[1]
            Location  = [[Folder1,RandImage1],[Folder1,RandImage2]]
            if Location not in InputByLocation:
                InputByLocation.append(Location)
                ImageLeft.append(self.Data[Folder1][RandImage1])
                ImageRight.append(self.Data[Folder1][RandImage2])
                Labels.append(1)
        
        for ii in range(8):
            IndexOfFolders = list(range(len(self.Data)))
            IndexOfFolders.remove(Folder1)
            _, Folder2 = np.array(random.sample(IndexOfFolders,k = 2))
            RandImages = np.array(random.sample(list(range(10)),k = 2))
            RandImage1 = RandImages[0]
            RandImage2 = RandImages[1]
            Location  = [[Folder1,RandImage1],[Folder2,RandImage2]]
            if Location not in InputByLocation:
                InputByLocation.append(Location)
                ImageLeft.append(self.Data[Folder1][RandImage1])
                ImageRight.append(self.Data[Folder2][RandImage2])
                if Folder1 == Folder2:
                    Labels.append(1)
                else:
                    Labels.append(0)
        ImageLeft,ImageRight, Labels = shuffle(ImageLeft,ImageRight,Labels,random_state=0)
        Pairs = [ImageLeft,ImageRight]
        return Pairs, Labels
    
    def Train(self, SaveModelPath = None):
        print('Start training')
        #print(len(Labels[0]))
        #Cache  = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype ='float32')
        #BestLoss = 0.2
        for  i  in range(self.Iteration):
            Pairs, Labels = self.GetBatch(i)     
            X = Pairs
            Y = np.array(Labels)
            Loss = self.Model.train_on_batch(X,Y)
            #self.TestOneShot()
            if SaveModelPath and i%1000 == 0:
                self.Model.save_weights(SaveModelPath)
                print('Epochs {} Loss: {}'.format(i,Loss))
                self.TestOneShot()
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
        Image1 = np.array([Image1], dtype='float32')
        Image1 = np.reshape(Image1,(1,124,124,1))
        Image2 = np.array([Image2], dtype='float32')
        Image2 = np.reshape(Image2,(1,124,124,1))
        return self.Model.predict_on_batch([Image1,Image2])

        
        
        


        

