'''
Created on 24 de abr de 2018

@author: maikon
'''
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

# cascade_src = 'resource/cars.xml'
cascade_src = '/home/maikon/git/OpencvPython/resource/haarcascade_russian_plate_number.xml'
car_cascade = cv2.CascadeClassifier(cascade_src)

def procurar_placa(frame, gray_image):
#     img    = cv2.imread('resource/placaFoco.jpg',0)
    img = frame.copy()
    
    newImg = cv2.blur(gray_image,(5,5))
    img    = newImg
    
    laplacian   = cv2.Laplacian(img,cv2.CV_64F)
    sobelx      = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=3,scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)
    sobely      = cv2.Sobel(img,cv2.CV_8U,0,1,ksize=3,scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)
    
    #aplicado o threshold sobre o Sobel de X
    tmp, imgThs = cv2.threshold(laplacian,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    
    #pequena chacoalhada nos pixels pra ver o que cai (isso limpa a img mas
    #distancia as regioes, experimente)
    #krl      = np.ones((6,6),np.uint8)
    #erosion  = cv2.erode(imgThs,krl,iterations = 1)
    #krl      = np.ones((19,19),np.uint8)
    #dilation = cv2.dilate(erosion,krl,iterations = 1) 
    #imgThs   = dilation
    
    #estrutura proporcional aa placa
    morph       = cv2.getStructuringElement(cv2.MORPH_RECT,(40,13))
    
    #captura das regioes que possam conter a placa
    plateDetect = cv2.morphologyEx(imgThs,cv2.MORPH_CLOSE,morph)
    regionPlate = plateDetect.copy()
    
    _, contours, hierarchy = cv2.findContours(regionPlate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    
    for contour in contours:
        [x,y,w,h] = cv2.boundingRect(contour)
     
        if h>250 and w>250:
            continue
     
        if h<40 or w<40:
            continue
     
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,255),2)
    
    cv2.drawContours(regionPlate,contours,-1,(255,255,255),18)
    cv2.imshow("Output - Press 'q' to exit", regionPlate)
    

def detectar_carro(frame, gray, car_cascade):
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        
      
cap = cv2.VideoCapture('resource/video.mp4')
# cap = cv2.VideoCapture('/home/maikon/Downloads/video1.avi')

img = cv2.imread("resource/meucarro.jpg")
g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
detectar_carro(img,g, car_cascade)
cv2.imshow("carro", img)
if not cap.isOpened():
    print("cannot open video")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
delay = int((1.0 / float(fps)) * 1000)

while (cap.isOpened()):
    ret, im = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
#     procurar_placa(im,gray)
    detectar_carro(im,gray, car_cascade)
    
    cv2.imshow("Orignal", im)
    k = cv2.waitKey(delay)
    if k & 0xFFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
   