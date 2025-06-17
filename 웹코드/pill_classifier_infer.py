import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json

# === 설정 ===
MODEL_PATH = "pill_resnet152_dataclass0_aug0.pt"
CLASS_JSON_PATH = "/Users/dotori/Downloads/AI 모델 소스코드/평가용 데이터셋/pill_data/pill_data_croped/pill_label_path_sharp_score_with_label.json"
IMAGE_DIR = "dir_testimage"
NUM_CLASSES = 1000  # 모델 학습 시 기준 클래스 수
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 클래스 매핑 로딩 ===
with open(CLASS_JSON_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)
    if isinstance(raw_data, list):
        id2label = {str(item[0]): item[1:] for item in raw_data}
    elif isinstance(raw_data, dict):
        id2label = raw_data
    else:
        raise ValueError("지원하지 않는 JSON 형식입니다.")

# === 이미지 전처리 ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === 이미지 판별 함수 ===
def is_image_file(filename):
    return filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))

# === 모델 로딩 함수 ===
def load_model(model_path):
    model = models.resnet152(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    checkpoint = torch.load(model_path, map_location=DEVICE)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.to(DEVICE)
    model.eval()
    return model

# === 예측 함수 ===
def predict_image(image_path, model):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[ERROR] {image_path} 이미지 열기 실패: {e}")
        return None

    image = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
        class_idx = pred.item()

    class_info = id2label.get(str(class_idx), None)
    if class_info:
        pill_id = class_info[0]
        pill_name = class_info[1]
        coords = class_info[2:] if len(class_info) > 2 else []
        coord_str = ", ".join([f"{c:.2f}" for c in coords])
        return f"{pill_name} ({pill_id})" + (f" | 좌표: {coord_str}" if coords else "")
    else:
        return f"Unknown({class_idx})"

# === 실행 ===
if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    print(f"[INFO] 모델 로딩 완료")

    for fname in os.listdir(IMAGE_DIR):
        fpath = os.path.join(IMAGE_DIR, fname)
        if os.path.isfile(fpath) and is_image_file(fname):
            result = predict_image(fpath, model)
            if result:
                print(f"[{fname}] → {result}")
        else:
            print(f"[SKIP] {fname} (디렉토리거나 이미지 아님)")
