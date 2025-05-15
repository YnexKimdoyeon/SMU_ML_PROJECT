import pandas as pd

# 원본 CSV 파일 읽기
df = pd.read_csv("결과파일.csv")  # 실제 파일명으로 변경하세요

# 랜덤으로 200,000개 샘플 추출
df_sampled = df.sample(n=200000, random_state=42)  # random_state는 결과 재현 가능하게 고정

# 새 파일로 저장
df_sampled.to_csv("random_200k.csv", index=False)

print("랜덤 20만 개 행이 'random_200k.csv'로 저장되었습니다.")
