import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import json
import ast

# ✅ CSV 파일에서 데이터 불러오기 (MODEL.ipynb에 사용된 파일명 기준)
data = pd.read_csv("data/final_data.csv", encoding='utf-8-sig')

# ✅ 질병명이 'non'인 경우 제거
data = data[data['질병명'] != 'non']

# ✅ 불필요한 열 제거
if '번호' in data.columns:
    data = data.drop(columns=['번호'])

# ✅ ATC 코드 vocabulary 생성
code_counter = Counter(
    code for _, row in data.iterrows()
    for code in ast.literal_eval(row['약품일반성분명코드(ATC코드)'])
)

# PAD: 0, UNK: 1로 고정 (추가)
voca = {"<PAD>": 0, "<UNK>": 1}
for idx, code in enumerate(code_counter, start=2):
    voca[code] = idx

# ✅ 질병명 라벨 인코딩
label_encoder = LabelEncoder()
data['질병명'] = label_encoder.fit_transform(data['질병명'])
label_map = {str(i): cls for i, cls in enumerate(label_encoder.classes_)}

# ✅ JSON으로 저장
with open("atc_vocab.json", "w", encoding="utf-8") as f:
    json.dump(voca, f, ensure_ascii=False, indent=2)

with open("label_map.json", "w", encoding="utf-8") as f:
    json.dump(label_map, f, ensure_ascii=False, indent=2)

print("✔️ atc_vocab.json 과 label_map.json 생성 완료")