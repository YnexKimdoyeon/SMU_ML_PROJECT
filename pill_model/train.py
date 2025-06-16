import os
import pandas as pd
from PIL import Image
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
        row = self.dataframe.iloc[idx]
        filename = row["filename"]
        label_name = row["label"]

        if filename not in os.listdir(self.image_dir):
            print(f"[파일 없음] {filename}")

        try:
            img_path = os.path.join(self.image_dir, filename)
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[에러] 이미지 로드 실패: {img_path} | 예외: {e}")
            return None

        if self.transform:
            image = self.transform(image)

        label = self.label2id.get(label_name)
        if label is None:
            print(f"[경고] 라벨 매핑 실패: {label_name}")
            return None

        return image, label

# === collate_fn: None 제거
def safe_collate(batch):
    batch = [x for x in batch if x is not None]
    return torch.utils.data.dataloader.default_collate(batch)

# === 경로 설정
image_dir = r"C:\Users\hotse\Downloads\166.약품식별 인공지능 개발을 위한 경구약제 이미지 데이터\01.데이터\1.Training\원천데이터\단일경구약제 5000종\filtered_images"
train_csv = "train_labels.csv"
val_csv = "val_labels.csv"

# === CSV 로드
df_train = pd.read_csv(train_csv, encoding="utf-8-sig")
df_val = pd.read_csv(val_csv, encoding="utf-8-sig")

# === 라벨 인덱싱
class_names = sorted(pd.concat([df_train, df_val])["label"].unique())
label2id = {label: idx for idx, label in enumerate(class_names)}
num_classes = len(label2id)

# === 전처리 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Dataset, Dataloader
train_dataset = PillDataset(df_train, image_dir, label2id, transform)
val_dataset = PillDataset(df_val, image_dir, label2id, transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=safe_collate)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=safe_collate)

# === 모델 정의
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# === 손실함수, 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# === 학습 루프
num_epochs = 10
best_val_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total
    print(f"[Epoch {epoch+1}] Train Loss: {running_loss:.4f} | Train Acc: {train_acc:.2f}%")

    # === 검증
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total
    print(f"          → Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # === best 모델 저장
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "pill_resnet152_best.pth")
        print(f"✅ Best model saved! (Val Acc: {val_acc:.2f}%)")

# === 최종 저장
torch.save(model.state_dict(), "pill_resnet152_final.pth")
print("✅ 최종 모델 저장 완료")
