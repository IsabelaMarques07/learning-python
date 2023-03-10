import cv2
from matplotlib import pyplot as plt
import numpy as np

img = cv2.imread('circulos.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# !branco
#image_lower_hsv = np.array([0, 1, 0])
#image_upper_hsv = np.array([180, 255, 255])

#máscara do contorno do vermelho 
image_lower_hsv = np.array([0, 165, 127])
image_upper_hsv = np.array([30, 255, 255])

mask_hsv = cv2.inRange(img_hsv, image_lower_hsv, image_upper_hsv)

#azul
image_lower_hsv2 = np.array([80, 165, 127])
image_upper_hsv2 = np.array([90, 255, 255])

mask_hsv2 = cv2.inRange(img_hsv, image_lower_hsv2, image_upper_hsv2)

#somamos as duas mascaras em uma imagem
mask_total = cv2.bitwise_or(mask_hsv, mask_hsv2)

mask_rgb = cv2.cvtColor(mask_total, cv2.COLOR_GRAY2RGB) 
contornos_img = mask_rgb.copy()
contornos, _ = cv2.findContours(mask_total, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

cv2.drawContours(contornos_img, contornos, -1, [255, 0, 0], 5)






plt.imshow(contornos_img)
plt.show()

#cont = []
#for contorno in contornos:
#    cont.append(cv2.contourArea(contorno))

#print(np.sort(cont))

#sortedarray = np.sort(cont)

#for contorno in contornos:
#    if(count == 1):
#        primeiroMaior = contorno
#    if(count == 2):
 #       segundoMaior = contorno
 #   else:



#plt.imshow(img[:,:,0], cmap="Greys_r", vmin=0, vmax=255); plt.show()
