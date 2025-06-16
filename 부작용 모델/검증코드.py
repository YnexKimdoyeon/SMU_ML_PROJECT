import joblib

# 저장된 모델 로드
model_data = joblib.load("model/full_model.pkl")
model = model_data['model']
le1 = model_data['le1']
le2 = model_data['le2']

# 예측 예시
drug1 = "Bicalutamide"
drug2 = "Calciportriol"

try:
    d1 = le1.transform([drug1])[0]
    d2 = le2.transform([drug2])[0]
    pred = model.predict([[d1, d2]])[0]
    print(f"💊 약물 쌍 ({drug1}, {drug2}) → 예측된 부작용 여부: {'있음' if pred == 1 else '없음'}")
except ValueError:
    print("❌ 입력한 약물이 인코더에 없습니다.")
