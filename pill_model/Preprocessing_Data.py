import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms

# 1. labels.csv 불러오기
df = pd.read_csv("labels.csv")  # 이전 단계에서 생성한 csv

# 2. 클래스 인덱싱
labels = sorted(df["label"].unique())  # 약 이름 알파벳 순 정렬
label2id = {name: i for i, name in enumerate(labels)}
id2label = {i: name for name, i in label2id.items()}

# 3. 라벨 숫자 인덱스로 변환
df["class_idx"] = df["label"].map(label2id)

# 4. 학습/검증 세트 분리 (8:2)
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df["class_idx"], random_state=42
)

# 5. 저장
train_df.to_csv("train_labels.csv", encoding="utf-8-sig", index=False)
val_df.to_csv("val_labels.csv", encoding="utf-8-sig", index=False)

# 6. 이미지 전처리 transform 정의 (torchvision용)
transform = transforms.Compose([
    transforms.Resize((224, 224)),            # 리사이즈
    transforms.ToTensor(),                    # [0, 1] 정규화
    transforms.Normalize([0.5, 0.5, 0.5],      # 평균
                         [0.5, 0.5, 0.5])      # 표준편차
])

print(f"✅ 전처리 완료: 클래스 수 = {len(label2id)}개")
