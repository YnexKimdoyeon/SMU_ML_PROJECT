import os
import zipfile
import json
import csv
import pandas as pd
from sklearn.model_selection import train_test_split

# === 0. 경로 설정 ===
label_zip_dir = r"C:\Users\hotse\Downloads\166.약품식별 인공지능 개발을 위한 경구약제 이미지 데이터\01.데이터\1.Training\라벨링데이터\단일경구약제 5000종"
filtered_image_dir = r"C:\Users\hotse\Downloads\166.약품식별 인공지능 개발을 위한 경구약제 이미지 데이터\01.데이터\1.Training\원천데이터\단일경구약제 5000종\filtered_images"
filtered_label_csv = "labels_existing.csv"
train_csv = "train_labels.csv"
val_csv = "val_labels.csv"

# === 1. filtered_images 내 실제 존재하는 이미지 파일명 수집 ===
filtered_image_name_set = set(os.listdir(filtered_image_dir))

# === 2. ZIP 파일 내 JSON을 열어 실제 존재하는 이미지만 필터링 ===
results = []
for i in range(1, 82):
    zip_path = os.path.join(label_zip_dir, f"TL_{i}_단일.zip")
    if not os.path.exists(zip_path):
        print(f"[!] ZIP 없음: TL_{i}_단일.zip")
        continue

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for fname in zf.namelist():
                if fname.endswith(".json"):
                    try:
                        with zf.open(fname) as f:
                            data = json.load(f)
                            if "images" in data:
                                for image_info in data["images"]:
                                    file_name = image_info.get("file_name", "")
                                    label = image_info.get("dl_name", "")
                                    if file_name in filtered_image_name_set:
                                        results.append((file_name, label))
                    except Exception as e:
                        print(f"[!] JSON 파싱 실패: {fname} → {e}")
    except zipfile.BadZipFile:
        print(f"[!] ZIP 손상됨: {zip_path}")
        continue

print(f"✅ 라벨링 포함된 유효 이미지 수: {len(results)}")

# === 3. 유효 이미지만 CSV로 저장 ===
with open(filtered_label_csv, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label"])
    writer.writerows(results)

print(f"✅ labels_existing_filtered.csv 저장 완료")

# === 4. DataFrame으로 읽어 stratified split ===
df = pd.read_csv(filtered_label_csv, encoding="utf-8-sig")

df_train, df_val = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

df_train.to_csv(train_csv, index=False, encoding="utf-8-sig")
df_val.to_csv(val_csv, index=False, encoding="utf-8-sig")

print(f"✅ 학습용 CSV 저장 완료 → train: {len(df_train)}개, val: {len(df_val)}개")
