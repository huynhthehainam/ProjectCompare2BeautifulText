from Model import SiameseModel
import cv2
import numpy as np
import os

os.system('cls')
#Image1 = cv2.imread('./Test/T1/Word0.png')
#Image2 = cv2.imread('./Test/T2/Word0.png')
model  = SiameseModel()
#model.LoadRawData('./Data')
model.LoadWeight('./WeightProject.h5')
#model.Train('WeightProject.h5')

count = 0
print('Label: 1')
for i in range(177):
	Image1 = cv2.imread('./Test/Result0/{}.png'.format(i),0)
	Image2 = cv2.imread('./Test/Result1/{}.png'.format(i),0)
	count = count + 1
	Pred = model.PredictOnePairImage(Image1,Image2)
	if Pred >= 0.5:
		count = count - 1
	else: 
		print('Image {}, Pred {}'.format(i,Pred))
#	print(Pred)

print('Label: 0')
for i in range(0,2):
    for ii in range(177-1):
        Image1 = cv2.imread('./Test/Result{}/{}.png'.format(i,ii),0)
        for iii in range(ii+1,5):
            Image2 = cv2.imread('./Test/Result{}/{}.png'.format(i,iii),0)
            count = count + 1
            Pred = model.PredictOnePairImage(Image1,Image2)
            if Pred < 0.5:
            	count = count - 1
            else:
                print('Pred {} Folder {} Image1 {} Image2 {}'.format(Pred,i,ii,iii))

'''
for i in range(177-1):
    Image1 = cv2.imread('./Test/Result0/{}.png'.format(i),0)
    Image2 = cv2.imread('./Test/Result1/{}.png'.format(i+1),0)
    count = count + 1
    Pred = model.PredictOnePairImage(Image1,Image2)
    if Pred < 0.5:
        nt  = count - 1
    #else:
        #print('Pred {} Folder {} Image1 {} Image2 {}'.format(Pred,i,ii,iii))
    #print(Pred)

for i in range(177-1):
    Image1 = cv2.imread('./Test/Result0/{}.png'.format(i),0)
    Image2 = cv2.imread('./Test/Result1/{}.png'.format(i+1),0)
    count = count + 1
    Pred = model.PredictOnePairImage(Image1,Image2)
    if Pred < 0.5:
        count = count - 1
    #else:
    #    print('Pred {} Folder {} Image1 {} Image2 {}'.format(Pred,i,ii,iii))
    #print(Pred)
'''
print ('loss: {}'.format(count))
#model.TestOneShot()for i in range(5-1):
# model.LoadWeight('./WeightProject.h5')
# Pred = model.PredictOnePairImage(Image1,Image2)
# print(Pred)