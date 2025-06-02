import pandas as pd
import os

csv_path = "labels_existing.csv"
image_dir = r"C:\Users\hotse\Downloads\166.약품식별 인공지능 개발을 위한 경구약제 이미지 데이터\01.데이터\1.Training\원천데이터\단일경구약제 5000종\filtered_images"

df = pd.read_csv(csv_path, encoding="utf-8-sig")
df = df[df["filename"].apply(lambda x: os.path.exists(os.path.join(image_dir, x)))]
df.to_csv("labels_existing_filtered.csv", index=False, encoding="utf-8-sig")

print(f"✅ 유효 이미지만 {len(df)}개 추려냄 → labels_existing_filtered.csv")
