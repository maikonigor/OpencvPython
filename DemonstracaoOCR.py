# -*- coding: utf-8 -*-
from PIL import Image
import pytesseract
import cv2

def removerChars(text):
    str = "!@#%¨&*()_+:;><^^}{`?|~¬\/=,.'ºª»‘"
    for x in str:
        text = text.replace(x, '')
    return text

video = cv2.VideoCapture(0)
# video = cv2.VideoCapture('resource/video.mp4')

while True:

    conect, frame = video.read()

    # area de localização
    area = frame[290:350, 150:450]

    # escala de cinza
    result = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)

    # limiarização
    ret, result = cv2.threshold(result, 40, 255, cv2.THRESH_BINARY)
    cv2.imshow('AREA', result)

    # desfoque
    result = cv2.GaussianBlur(result, (5, 5), 0)
    cv2.imshow('DESFOQUE', result)

    # limite horizontal
    cv2.line(frame, (0, 250), (640, 250), (0, 0, 255), 2)
    # limite horizonta2
    cv2.line(frame, (0, 370), (640, 370), (0, 0, 255), 2)
    # limite vertical1
    cv2.line(frame, (130, 0), (130, 480), (0, 0, 255), 1)
    # limite vertical2
    cv2.line(frame, (470, 0), (470, 480), (0, 0, 255), 1)

    cv2.imwrite("demo/roi.jpg", result)
    cv2.imshow("image",result)
    
    imagem = Image.open("./demo/roi.jpg")
    saida = pytesseract.image_to_string(imagem, lang='eng')
    txt = removerChars(saida)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, txt, (180, 50), font, 2, (255, 255, 255), 2, cv2.LINE_AA)


    cv2.imshow('FRAME', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

video.release()
cv2.destroyAllWindows()