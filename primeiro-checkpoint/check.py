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

#mostra as imagens
plt.imshow(contornos_img)
plt.show()