import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# === 사용자 정의 Dataset ===
class PillDataset(Dataset):
    def __init__(self, dataframe, image_dir, label2id, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.label2id = label2id
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]["filename"]
        label_name = self.dataframe.iloc[idx]["label"]
        img_path = os.path.join(self.image_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = self.label2id[label_name]
        return image, label

# === 경로 설정 ===
csv_path = "labels_existing_filtered.csv"
df = pd.read_csv(csv_path, encoding="utf-8-sig")  # 한글 깨짐 방지
image_dir = r"C:\Users\hotse\Downloads\166.약품식별 인공지능 개발을 위한 경구약제 이미지 데이터\01.데이터\1.Training\원천데이터\단일경구약제 5000종\filtered_images"

# === CSV 로드 및 클래스 인덱싱 ===
df = pd.read_csv(csv_path)
class_names = sorted(df["label"].unique())
label2id = {name: idx for idx, name in enumerate(class_names)}
num_classes = len(class_names)

# === 데이터 분할 ===
df_train, df_val = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

# === 전처리 ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === Dataset 및 Dataloader 생성 ===
train_dataset = PillDataset(df_train, image_dir, label2id, transform)
val_dataset = PillDataset(df_val, image_dir, label2id, transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# === 모델 정의 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# === 손실함수 및 옵티마이저 ===
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# === 학습 루프 ===
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    print(f"[Epoch {epoch+1}] Loss: {running_loss:.4f} | Train Acc: {acc:.2f}%")
    
# === 검증 데이터셋 평가 ===
model.eval()
val_loss = 0.0
val_correct = 0
val_total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        val_correct += (preds == labels).sum().item()
        val_total += labels.size(0)
        
# === 모델 저장 ===
torch.save(model.state_dict(), "pill_resnet152.pth")
print("✅ 학습 완료 및 모델 저장 완료")
