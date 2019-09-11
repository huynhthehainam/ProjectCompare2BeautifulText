import cv2
import os
import glob
os.system('cls')

files = glob.glob('./Result/*')
for f in files:
    os.remove(f)
T = 175
Img=  cv2.imread('Capture1_2.PNG')

Gray  = cv2.cvtColor(src = Img,code= cv2.COLOR_BGR2GRAY)

# Blur = cv2.GaussianBlur(src = Gray, ksize=(5,5), sigmaX=0)
(T, Binary) = cv2.threshold(src = Gray,thresh=T,maxval=255,type=cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
RectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
Dilation = cv2.dilate(Binary, RectKernel, iterations =1)


Contours, hierarchy = cv2.findContours(image= Dilation, mode=cv2.RETR_TREE,method=cv2.CHAIN_APPROX_SIMPLE)
for Contour in Contours:
    x,y,w,h = cv2.boundingRect(Contour)
    print(x,y,w,h)
    Img = cv2.rectangle(Img, (x, y), (x+w, y+h), color = (255, 0, 0), thickness = 1)

cv2.imwrite('Result.png',Img)