import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import os

# 1. 데이터 불러오기
df = pd.read_csv("replaced_interaction_data.csv")

# 2. 부작용 라벨링
def label_target(x):
    x = str(x).strip().upper()
    return 1 if x != 'X' and x != '' else 0

df['target'] = df['부작용'].apply(label_target)

# 3. 약물1, 약물2 쌍 뒤집기
df_swapped = df.copy()
df_swapped['약물1'], df_swapped['약물2'] = df['약물2'], df['약물1']

# 4. 원본 + 뒤집은 데이터 병합 및 중복 제거
df_aug = pd.concat([df, df_swapped], ignore_index=True).drop_duplicates()

# 5. 인코딩
le1 = LabelEncoder()
le2 = LabelEncoder()
df_aug['약물1_enc'] = le1.fit_transform(df_aug['약물1'])
df_aug['약물2_enc'] = le2.fit_transform(df_aug['약물2'])

X = df_aug[['약물1_enc', '약물2_enc']]
y = df_aug['target']

# 6. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 7. SMOTE로 클래스 불균형 보정
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 8. LightGBM 학습
model = LGBMClassifier(
    num_leaves=64,
    max_depth=7,
    learning_rate=0.05,
    n_estimators=200,
    random_state=42
)
model.fit(X_train_res, y_train_res)

# 9. 평가
y_pred = model.predict(X_test)
print("📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📄 Classification Report:\n", classification_report(y_test, y_pred))

# 10. 저장
os.makedirs("model", exist_ok=True)
joblib.dump({
    "model": model,
    "le1": le1,
    "le2": le2
}, "model/full_model.pkl")

print("✅ 모델이 model/full_model.pkl 로 저장되었습니다.")
