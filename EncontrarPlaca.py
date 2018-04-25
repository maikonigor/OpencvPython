# -*- coding: utf-8 -*-
#######################################################
#     Detecção de Placas atraves de contornos         #
#                       by AY7                        #
#######################################################

from PIL import Image
import numpy as np
# import tkinter
import pytesseract
import cv2

def desenhaContornos(contornos, imagem):
    for c in contornos:
        #perimetro do contorno, verifica se o contorno é fechado
        perimetro = cv2.arcLength(c, True)
        if perimetro > 80:
           #aproxima os contornos da forma correspondente
           approx = cv2.approxPolyDP(c, 0.03 * perimetro, True)
           #verifica se é um quadrado ou retangulo de acordo com a qtd de vertices
           if len(approx) == 4:
               #cv2.drawContours(imagem, [c], -1, (0, 255, 0), 1)
                (x, y, a, l) = cv2.boundingRect(c)
                cv2.rectangle(imagem, (x, y), (x + a, y + l), (0, 255, 0), 2)
                roi = imagem[y:y + l, x:x + a]
                cv2.imwrite("demo/roi.jpg", roi)

    return imagem

    # Captura ou Video

def reconhecimentoOCR(path_img):
    entrada = cv2.imread(path_img + ".jpg")
    # cv2.imshow("ENTRADA", img)

    # amplia a imagem da placa em 4
    img = cv2.resize(entrada, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    # Converte para escala de cinza
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Escala Cinza", img)

    # Binariza imagem
    ret, img = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
    cv2.imshow("Limiar", img)

    # Desfoque na Imagem
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # cv2.imshow("Desfoque", img)

    cv2.imwrite(path_img + "-ocr.jpg", img)
    imagem = Image.open(path_img + "-ocr.jpg")
    saida = pytesseract.image_to_string(imagem, lang='eng')
    print(saida)
    
    texto = removerChars(saida)
    print(texto)
#     janela = tkinter.Tk()
#     tkinter.Label(janela, text=texto, font=("Helvetica", 50)).pack()
#     janela.mainloop()

    # cv2.waitKey(0)
    cv2.destroyAllWindows()

def removerChars(self, text):
    str = "!@#%¨&*()_+:;><^^}{`?|~¬\/=,.'ºª»‘"
    for x in str:
        text = text.replace(x, '')
    return text

# video = cv2.VideoCapture('resource\\video1-720p-edit2.mp4')
video = cv2.VideoCapture(0)
while True:

    ret, frame = video.read()

    # limite horizontal
    cv2.line(frame, (0, 350), (860, 350), (0, 0, 255), 1)
    # limite vertical 1
    cv2.line(frame, (220, 0), (220, 480), (0, 0, 255), 1)
    # limite vertical 2
    cv2.line(frame, (500, 0), (500, 480), (0, 0, 255), 1)

    cv2.imshow('SAIDA', frame)

    # região de busca
    res = frame[350:, 220:500]

    # escala de cinza
    img_result = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    # limiarização
    ret, img_result = cv2.threshold(img_result, 90, 255, cv2.THRESH_BINARY)

    # desfoque
    img_result = cv2.GaussianBlur(img_result, (5, 5), 0)

    # lista os contornos
    img, contornos, hier = cv2.findContours(img_result, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    desenhaContornos(contornos, res)

    cv2.imshow('RES', res)

    if cv2.waitKey(1) == ord('q'):
        break

reconhecimentoOCR("demo/roi")

video.release()
cv2.destroyAllWindows()



