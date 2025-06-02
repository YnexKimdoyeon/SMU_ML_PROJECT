import pandas as pd
from sklearn.model_selection import train_test_split

# CSV 로드
df = pd.read_csv("labels_existing_filtered.csv", encoding="utf-8-sig")

# 클래스당 최소 2개 이상 유지
counts = df["label"].value_counts()
valid_labels = counts[counts >= 2].index
df_filtered = df[df["label"].isin(valid_labels)].reset_index(drop=True)

num_classes = df_filtered["label"].nunique()
num_samples = len(df_filtered)
val_size = int(num_samples * 0.2)

print(f"클래스 수: {num_classes}, 전체 샘플 수: {num_samples}, val셋 예상 크기: {val_size}")

# 조건 확인
if val_size >= num_classes:
    # stratify 사용 가능
    df_train, df_val = train_test_split(
        df_filtered,
        test_size=0.2,
        stratify=df_filtered["label"],
        random_state=42
    )
    print("✅ stratify 기반 분할 수행")
else:
    # stratify 사용 불가 → 랜덤 분할
    df_train, df_val = train_test_split(
        df_filtered,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )
    print("⚠️ stratify 불가 → 랜덤 분할로 대체")

# 저장
df_train.to_csv("train_labels.csv", index=False, encoding="utf-8-sig")
df_val.to_csv("val_labels.csv", index=False, encoding="utf-8-sig")
print(f"✅ train {len(df_train)}개 / val {len(df_val)}개 저장 완료")
