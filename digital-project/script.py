!#python38

import cv2
import mediapipe as mp

#OCR Biblioteca para reconhecer números, caracteres

imagem = cv2.imread('caminho/para/imagem.jpg')

mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1)
