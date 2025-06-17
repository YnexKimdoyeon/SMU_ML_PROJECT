from pill_classifier import valid
from get_cli_args import get_cli_args
from torchvision import models, transforms
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# === Dataset 정의 (정답 포함)
class ValidationDataset(Dataset):
    def __init__(self, dataframe, image_dir, label2id, transform=None):
        self.df = dataframe
        self.image_dir = image_dir
        self.label2id = label2id
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")
        label = self.label2id[row["label"]]

        if self.transform:
            image = self.transform(image)

        return image, label, row["filename"], ""

# === transform 정의
transform_normalize = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === 설정
args = get_cli_args(job='resnet152', run_phase='valid', aug_level=0, dataclass='01')
df_valid = pd.read_csv("valid_labels.csv")  # ← 실제 라벨 파일 필요
label2id = {label: idx for idx, label in enumerate(sorted(df_valid["label"].unique()))}

# === 데이터셋 & 로더
image_dir = "dir_testimage"
dataset_valid = ValidationDataset(df_valid, image_dir, label2id, transform=transform_normalize)
dataloader_valid = DataLoader(dataset_valid, batch_size=32, shuffle=False)

# === 모델 정의
model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, len(label2id))
model.load_state_dict(torch.load(args.model_path_in))  # 모델 가중치 로드
model = model.cuda() if args.cuda else model

# === 손실함수
criterion = nn.CrossEntropyLoss()

# === 검증 수행
print("검증 시작")
valid(args, dataloader_valid, sampler=None, model=model, criterion=criterion, epoch=0)
print("검증 완료")
