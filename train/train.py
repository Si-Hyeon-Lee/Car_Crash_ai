import glob
import json
import os
from PIL import Image, ImageDraw
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CarDamageDataset
from torchmetrics import BinaryJaccardIndex
from Models import Segmentation_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 2
model = Segmentation_model(num_classes=num_classes).to(device)

pos_weight = torch.tensor([10]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # For binary classification, adjust if necessary
optimizer = optim.Adam(model.parameters(), lr=0.001)
# BinaryJaccardIndex 인스턴스 생성
binary_jaccard_index = BinaryJaccardIndex().to(device)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset = CarDamageDataset(json_dir='./car_data/all/json',
                           img_dir='./car_data/all/img',
                           transform=transform)
data_loader = DataLoader(dataset, batch_size=128, shuffle=True)

val_dataset =CarDamageDataset(json_dir = './car_data/val/json',
                            img_dir = './car_data/val/img',
                            transform = transform)
val_data_loader = DataLoader(val_dataset,batch_size = 128,shuffle=False)

num_epochs = 100
save_dir = './car_data/models_save/'

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    # Validation Loop
    if epoch % 10 == 9:
        total_jaccard_index = 0.0
        count = 0
        model.eval()
        with torch.no_grad():
            for idx, (images, masks) in val_data_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                preds = torch.sigmoid(outputs) > 0.5
                preds = preds.float() # BinaryJaccardIndex를 위해 preds를 float로 변경

                # BinaryJaccardIndex 계산
                score = binary_jaccard_index(preds, masks.float()) # masks도 float으로 변경
                total_jaccard_index += score.item() #스칼라
                count += 1

            mean_jaccard_index = total_jaccard_index / count
            print(f"Epoch {epoch+1}, mIOU: {mean_jaccard_index:.4f}")
            torch.save(model.state_dict(), os.path.join(save_dir, f'{epoch+1}_model.pth'))

    for images, masks in data_loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(data_loader.dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}')