import os
import zipfile
import json
import csv

# 1. 라벨링 ZIP 경로
label_zip_dir = r"C:\Users\hotse\Downloads\166.약품식별 인공지능 개발을 위한 경구약제 이미지 데이터\01.데이터\1.Training\라벨링데이터\단일경구약제 5000종"
# 2. 원천 ZIP 경로
image_zip_dir = r"C:\Users\hotse\Downloads\166.약품식별 인공지능 개발을 위한 경구약제 이미지 데이터\01.데이터\1.Training\원천데이터\단일경구약제 5000종"

# 3. 결과 저장할 CSV
output_csv = "labels_existing.csv"
results = []

# 4. ZIP 이름은 TL_xx_단일.zip, TS_xx_단일.zip
for i in range(1, 82):
    label_zip_path = os.path.join(label_zip_dir, f"TL_{i}_단일.zip")
    image_zip_path = os.path.join(image_zip_dir, f"TS_{i}_단일.zip")

    if not os.path.exists(label_zip_path):
        print(f"[!] 라벨링 ZIP 없음: TL_{i}_단일.zip")
        continue
    if not os.path.exists(image_zip_path):
        print(f"[!] 원천 ZIP 없음: TS_{i}_단일.zip → 해당 라벨 무시")
        continue

    # 1. 원천 ZIP 안의 이미지 이름 목록 추출
    image_name_set = set()
    with zipfile.ZipFile(image_zip_path, 'r') as izip:
        for fname in izip.namelist():
            if fname.endswith(".png") or fname.endswith(".jpg"):
                image_name_set.add(os.path.basename(fname))

    # 2. 라벨 ZIP 안의 JSON에서 image_name → dl_name 추출
    with zipfile.ZipFile(label_zip_path, 'r') as lzip:
        for fname in lzip.namelist():
            if fname.endswith(".json"):
                try:
                    with lzip.open(fname) as f:
                        data = json.load(f)
                        image_info = data.get("images", [{}])[0]
                        file_name = image_info.get("file_name", "")
                        label = image_info.get("dl_name", "")

                        # 실제 이미지가 존재할 경우만 포함
                        if file_name in image_name_set:
                            results.append((file_name, label))
                except Exception as e:
                    print(f"[!] JSON 읽기 실패 ({fname}): {e}")

print(f"총 유효한 항목 수: {len(results)}")

# 5. CSV 저장
with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label"])
    writer.writerows(results)

print(f"✅ 유효 이미지 기준 labels_existing.csv 저장 완료")
