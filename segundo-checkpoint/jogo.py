import cv2
from matplotlib import pyplot as plt
import numpy as np

#imagem utilizada
img_pedra = cv2.imread('pedra.png')
img_papel = cv2.imread('papel.png')
img_tesoura = cv2.imread('tesoura.png')

img_pedra_rgb = cv2.cvtColor(img_pedra, cv2.COLOR_BGR2RGB)
img_papel_rgb = cv2.cvtColor(img_papel, cv2.COLOR_BGR2RGB)
img_tesoura_rgb = cv2.cvtColor(img_tesoura, cv2.COLOR_BGR2RGB)

#KEYPOINTS
orb = cv2.ORB_create()
#calcular os keypoints das imagens

kp_pedra, des_pedra = orb.detectAndCompute(img_pedra_rgb,None)
kp_papel, des_papel = orb.detectAndCompute(img_papel_rgb,None)
kp_tesoura, des_tesoura = orb.detectAndCompute(img_tesoura_rgb,None)
# Desenha os keypoints na imagem 
pedra_kp_img = cv2.drawKeypoints(img_pedra_rgb, kp_pedra, outImage=np.array([]), flags=0)

#carrega o video 
video = cv2.VideoCapture('pedra-papel-tesoura.mp4')

# Cria a subtração do fundo
#fgbg = cv2.createBackgroundSubtractorMOG2()
#fgbg = cv2.createBackgroundSubtractorKNN()
#
#while(1):
#    ret, frame = video.read()
#    
#    if not ret:
#        break
#    
#    # Aplica a mascara no frame recebido
#    #fgmask = fgbg.apply(frame)
#    video_resized = cv2.resize(frame,(1280,720),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
#    cv2.imshow('fgmask',video_resized)
#    #cv2.imshow('frame',fgmask)
#
#    
#    k = cv2.waitKey(30) & 0xff
#    if k == 27:
#        break
#    
#
#video.release()
#cv2.destroyAllWindows()

#cv2.imshow(pedra_kp_img)

#mostra as imagens
plt.imshow(pedra_kp_img)
plt.show()
