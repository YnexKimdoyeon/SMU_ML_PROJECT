import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd

# === 클래스 정보 로드 ===
csv_path = "labels_existing_filtered.csv"
df = pd.read_csv(csv_path, encoding="utf-8-sig")
class_names = sorted(df["label"].unique())
id2label = {idx: name for idx, name in enumerate(class_names)}
num_classes = len(class_names)

# === 모델 정의 및 로드 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet152(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("pill_resnet152.pth", map_location=device))
model = model.to(device)
model.eval()

# === 이미지 전처리 정의 ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === 이미지 경로 인자 받아서 처리 ===
if len(sys.argv) < 2:
    print("이미지 경로를 인자로 넣어주세요.")
    sys.exit(1)

image_path = sys.argv[1]
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)  # 배치 차원 추가

# === 예측 ===
with torch.no_grad():
    outputs = model(image)
    _, pred = torch.max(outputs, 1)
    predicted_label = id2label[pred.item()]
    print(f"✅ 예측 결과: {predicted_label}")
