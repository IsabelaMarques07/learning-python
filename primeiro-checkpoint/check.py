#import das dependências
import math
import cv2
from matplotlib import pyplot as plt
import numpy as np

#mostra as imagens
#plt.imshow(contornos_img)
#plt.show()

def image_da_webcam(img):
    """
    ->>> !!!! FECHE A JANELA COM A TECLA ESC !!!! <<<<-
        deve receber a imagem da camera e retornar uma imagems filtrada.
    """
    #imagem utilizada
    img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #img = cv2.imread('circulos.png')

    #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #máscara do contorno do vermelho 
    image_lower_hsv = np.array([0, 180, 160])
    image_upper_hsv = np.array([40, 255, 255])
    mask_hsv = cv2.inRange(img_hsv, image_lower_hsv, image_upper_hsv)
    mask_hsv = cv2.erode(mask_hsv, None, iterations=20)
    mask_hsv = cv2.dilate(mask_hsv, None, iterations=20)

    #máscara do contorno do amarelo
    image_lower_hsv2 = np.array([20, 230, 250])
    image_upper_hsv2 = np.array([60, 255, 255])
    mask_hsv2 = cv2.inRange(img_hsv, image_lower_hsv2, image_upper_hsv2)
    mask_hsv2 = cv2.erode(mask_hsv2, None, iterations=20)
    mask_hsv2 = cv2.dilate(mask_hsv2, None, iterations=20)

    #somamos as duas mascaras em uma única imagem
    mask_total = cv2.bitwise_or(mask_hsv, mask_hsv2)

    #passa a máscara para rgb
    mask_rgb = cv2.cvtColor(mask_total, cv2.COLOR_GRAY2RGB)
    #cria uma cópia da imagem com a máscara
    contornos_img = mask_rgb.copy()
    #acha os contornos da imagem
    contornos, _ = cv2.findContours(mask_total, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    #desenha os contornos na imagem
    cv2.drawContours(contornos_img, contornos, -1, [255, 0, 0], 5)

    coordenadasX = []
    coordenadasY = []
    for contorno in contornos:
        #tamanho e cor da cruz
        size = 20
        color = (115,251,253)


        M = cv2.moments(contorno)

        if (M['m00'] != 0):
            #centro das formas
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            coordenadasX.append(cx)
            coordenadasY.append(cy)
            #desenhar a cruz
            cv2.line(img,(cx - size,cy),(cx + size,cy),color,5)
            cv2.line(img,(cx,cy - size),(cx, cy + size),color,5)

            #desenha contornos
            rect = cv2.minAreaRect(contorno)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], -1, [255, 0, 0], 3)


            # Escrever a massa
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = cy , cx
            origem_posicao = (cx - 105,cy - 90)
            origem_area = (cx - 105,cy - 110)
            area = cv2.contourArea(contorno)
            cv2.putText(img, "posicao: "+str(text), origem_posicao, font,0.65,(200,50,0),1,cv2.LINE_AA)
            cv2.putText(img, "area: "+str(area), origem_area, font,0.65,(200,50,0),1,cv2.LINE_AA)
            print(len(coordenadasX))
            if (len(coordenadasX) > 1):
                print("aqui")
                color = (250,120,230)
                cv2.line(img,(coordenadasX[0],coordenadasY[0]),(coordenadasX[1], coordenadasY[1]),color,5)
                #calcular o ângulo de inclinação
                dx = coordenadasX[0] - coordenadasX[1]
                dy = coordenadasY[0] - coordenadasY[1]

                rad = math.atan2(dy, dx)
                angulo = math.degrees(rad)
                posicao = (10, 20)
                if angulo < 0:
                    angulo += 360
                    
                cv2.putText(img, "angulo: "+str(angulo), posicao, font,0.6,(200,50,0),1,cv2.LINE_AA)

    return img

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)


if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    
    img = image_da_webcam(frame)


    cv2.imshow("preview", img)

    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

cv2.destroyWindow("preview")
vc.release()