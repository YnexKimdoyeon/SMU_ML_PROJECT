import torch
import torch.nn as nn
from torchvision import models

from Preprocessing_Data import label2id

# 클래스 수 (전처리 단계에서 생성된 label2id 사용)
num_classes = len(label2id)

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ResNet-152 불러오기
model = models.resnet152(pretrained=True)

# 마지막 FC layer 수정
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 장치로 이동
model = model.to(device)

print("✅ 모델 구성 완료: ResNet-152 →", num_classes, "개 클래스 분류")
