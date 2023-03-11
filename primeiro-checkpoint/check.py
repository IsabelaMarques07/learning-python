#import das dependências
import cv2
from matplotlib import pyplot as plt
import numpy as np

#imagem utilizada
img = cv2.imread('circulos.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#máscara do contorno do vermelho 
image_lower_hsv = np.array([0, 165, 127])
image_upper_hsv = np.array([30, 255, 255])
mask_hsv = cv2.inRange(img_hsv, image_lower_hsv, image_upper_hsv)

#máscara do contorno do azul
image_lower_hsv2 = np.array([80, 165, 127])
image_upper_hsv2 = np.array([90, 255, 240])
mask_hsv2 = cv2.inRange(img_hsv, image_lower_hsv2, image_upper_hsv2)

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

#tamanho e cor da cruz
size = 20
color = (115,251,253)
count = 0
for contorno in contornos:
    cnt = contornos[count]
    M = cv2.moments(cnt)
    #centro das formas
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    #desenhar a cruz
    cv2.line(contornos_img,(cx - size,cy),(cx + size,cy),color,5)
    cv2.line(contornos_img,(cx,cy - size),(cx, cy + size),color,5)
    # Escrever a massa
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = cy , cx
    origem = (cx - 100,cy - 100)
    cv2.putText(contornos_img, str(text), origem, font,1,(200,50,0),2,cv2.LINE_AA)
    count += 1


#mostra as imagens
plt.imshow(contornos_img)
plt.show()