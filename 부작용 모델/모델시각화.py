import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

# 1. 저장된 모델 로드
model_bundle = joblib.load("model/full_model.pkl")
model = model_bundle["model"]
le1 = model_bundle["le1"]
le2 = model_bundle["le2"]

# 2. 테스트용 데이터 로드 및 전처리
df = pd.read_csv("replaced_interaction_data.csv")

def label_target(x):
    x = str(x).strip().upper()
    return 1 if x != 'X' and x != '' else 0

df['target'] = df['부작용'].apply(label_target)

df_swapped = df.copy()
df_swapped['약물1'], df_swapped['약물2'] = df['약물2'], df['약물1']
df_aug = pd.concat([df, df_swapped], ignore_index=True).drop_duplicates()

df_aug['약물1_enc'] = le1.transform(df_aug['약물1'])
df_aug['약물2_enc'] = le2.transform(df_aug['약물2'])

X = df_aug[['약물1_enc', '약물2_enc']]
y = df_aug['target']

# 3. 예측
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

# 4. 평가 출력
print("📊 Confusion Matrix:\n", confusion_matrix(y, y_pred))
print("\n📄 Classification Report:\n", classification_report(y, y_pred))

# 5. ROC Curve
fpr, tpr, _ = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

# 6. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y, y_prob)
plt.figure()
plt.plot(recall, precision, label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid()
plt.show()
