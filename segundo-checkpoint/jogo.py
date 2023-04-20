import numpy as np
import cv2

# Carrega as imagens modelo
pedra_img = cv2.imread("pedra.png", cv2.IMREAD_GRAYSCALE)
papel_img = cv2.imread("papel.png", cv2.IMREAD_GRAYSCALE)
tesoura_img = cv2.imread("tesoura.png", cv2.IMREAD_GRAYSCALE)

# Configura o detector SIFT
sift = cv2.SIFT_create()

# Encontra os keypoints e descritores das imagens modelo
kp1, des1 = sift.detectAndCompute(pedra_img, None)
kp2, des2 = sift.detectAndCompute(papel_img, None)
kp3, des3 = sift.detectAndCompute(tesoura_img, None)

MIN_MATCHES = 10

# Inicializa o objeto de captura de vídeo
cap = cv2.VideoCapture("pedra-papel-tesoura.mp4")

player1_wins = 0
player2_wins = 0
player1_hand = ''
player2_hand = ''

#FUNÇÃO QUE RECEBE OS MATCHES E VERIFICA O SINAL DE CADA JOGADOR
def reconhecer_jogadores(good_tesoura, good_pedra, good_papel, frame, player1_hand, player2_hand):
    old_hand1 = player1_hand
    old_hand2 = player2_hand
    # Se houver correspondências suficientes, mostra o nome da imagem correspondente
    if len(good_pedra) > MIN_MATCHES: 
        if len(good_papel) > MIN_MATCHES or len(good_tesoura) > MIN_MATCHES:
            if kp[good_pedra[0].trainIdx].pt[0] < gray.shape[1] / 2:
                    player1_hand = 'rock'
                    cv2.putText(frame, "Jogador 1: Pedra", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 255), 2)
            else:
                    player2_hand = 'rock'
                    cv2.putText(frame, "Jogador 2: Pedra", (1000, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
        else:
             player1_hand = 'rock'
             cv2.putText(frame, "Ambos: Pedra", (120, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
             player2_hand = 'rock'

    if len(good_papel) > MIN_MATCHES:
        if len(good_tesoura) > MIN_MATCHES or len(good_pedra) > MIN_MATCHES:
            if kp[good_papel[0].trainIdx].pt[0] < gray.shape[1] / 2:
                    player1_hand = 'paper'
                    cv2.putText(frame, "Jogador 1: Papel", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 255), 2)
            else:
                    player2_hand = 'paper'
                    cv2.putText(frame, "Jogador 2: Papel", (1000, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
        else:
             player1_hand = 'papel'
             cv2.putText(frame, "Ambos: Papel", (120, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
             player2_hand = 'papel'

    if len(good_tesoura) > MIN_MATCHES:
        if len(good_pedra) > MIN_MATCHES or len(good_papel) > MIN_MATCHES:
            if kp[good_tesoura[0].trainIdx].pt[0] < gray.shape[1] / 2:
                    player1_hand = 'scissors'
                    cv2.putText(frame, "Jogador 1: Tesoura", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 255), 2)
            else:
                    player2_hand = 'scissors'
                    cv2.putText(frame, "Jogador 2: Tesoura", (1000, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
        else:
             player1_hand = 'tesoura'
             cv2.putText(frame, "Ambos: Tesoura", (120, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
             player2_hand = 'tesoura'
    
    return [player1_hand, player2_hand, frame, old_hand1, old_hand2]

#FUNÇÃO QUE RECEBE OS SINAIS DE CADA JOGADOR E RETORNA QUAL GANHOU OU SE HOUVE EMPATE
def verificar_ganhador(player1_hand, player2_hand, player1_wins, player2_wins):
    mensagem = ''
    # Verificar quem ganhou a rodada
    if player1_hand == 'rock' and player2_hand == 'scissors' or \
        player1_hand == 'paper' and player2_hand == 'rock' or \
        player1_hand == 'scissors' and player2_hand == 'paper':
        player1_wins += 1
        mensagem = "Jogador 1 ganhou!"
    elif player2_hand == 'rock' and player1_hand == 'scissors' or \
            player2_hand == 'paper' and player1_hand == 'rock' or \
            player2_hand == 'scissors' and player1_hand == 'paper':
        cv2.putText(frame, "Jogador 2 ganhou!", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        player2_wins += 1
        mensagem = "Jogador 2 ganhou!"
    else:
        cv2.putText(frame, "Empate!", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        mensagem = "Empate!"
    
    return [frame, player1_wins, player2_wins, mensagem]

count = 0
while(cap.isOpened()):
    # Lê um frame do vídeo a cada 20 frames
    ret, frame = cap.read()
    count += 1
    if count % 10 != 0 or not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    bf = cv2.BFMatcher()

    matches1 = bf.knnMatch(des1, des, k=2)
    matches2 = bf.knnMatch(des2, des, k=2)
    matches3 = bf.knnMatch(des3, des, k=2)

    good_pedra = []
    good_papel = []
    good_tesoura = []
    for m, n in matches1:
        if m.distance < 0.75 * n.distance:
            good_pedra.append(m)
    for m, n in matches2:
        if m.distance < 0.75 * n.distance:
            good_papel.append(m)
    for m, n in matches3:
        if m.distance < 0.75 * n.distance:
            good_tesoura.append(m) 

    [player1_hand, player2_hand, frame, old_hand1, old_hand2] = reconhecer_jogadores(good_tesoura, good_pedra, good_papel, frame, player1_hand, player2_hand)

    if(old_hand1 != player1_hand or old_hand2 != old_hand2):
        [frame, player1_wins, player2_wins, mensagem] = verificar_ganhador(player1_hand, player2_hand, player1_wins, player2_wins)

    cv2.putText(frame, mensagem, (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, "Jogador1: {}".format(player1_wins), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, "Jogador2: {}".format(player2_wins), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)

    # Se a tecla 'q' for pressionada, encerra o loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
