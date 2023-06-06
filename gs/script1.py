import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import cv2

transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

cap = cv2.VideoCapture("video.mp4")
train_dataset = ImageFolder('dataset/train', transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=30, shuffle=True)

model = nn.Sequential(
    nn.Linear(100 * 100 * 3, 64), 
    nn.ReLU(),
    nn.Linear(64, 2),
    nn.Softmax(dim=1)
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for images, labels in train_dataloader:
        images = images.view(images.size(0), -1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{10}, Loss: {loss.item()}")

unknown_image = Image.open('podre_13.jpg')
unknown_image = transform(unknown_image).unsqueeze(0) 

model.eval()
with torch.no_grad():
    unknown_image = unknown_image.view(unknown_image.size(0), -1)
    outputs = model(unknown_image)
    _, predicted = torch.max(outputs, 1)


class_mapping = {0: "saudavel", 1: "podre"}
predicted_class = class_mapping[predicted.item()]
print("A imagem desconhecida é provavelmente uma semente", predicted_class)



#     def analyze_frame(frame):

#         unknown_image = frame
#         unknown_image = transform(unknown_image).unsqueeze(0) 

#         model.eval()  # Alterar o modo do modelo para avaliação
#         with torch.no_grad():
#             unknown_image = unknown_image.view(unknown_image.size(0), -1)
#             outputs = model(unknown_image)
#             _, predicted = torch.max(outputs, 1)


#         class_mapping = {0: "saudavel", 1: "podre"}
#         predicted_class = class_mapping[predicted.item()]
#         print("A imagem desconhecida é provavelmente uma semente", predicted_class)
#         return predicted_class



# while(cap.isOpened()):
#     # Lê um frame do vídeo a cada 20 frames
#     ret, frame = cap.read()
#     if not ret:
#         continue

#     frame_pil = Image.fromarray(frame)
#     mensagem = analyze_frame(frame_pil)

#     cv2.putText(frame, mensagem, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     cv2.imshow("Frame", frame)

#     # Se a tecla 'q' for pressionada, encerra o loop
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break