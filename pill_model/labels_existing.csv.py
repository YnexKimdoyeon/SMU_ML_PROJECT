import os
import zipfile
import json
import csv

# 1. 라벨링 ZIP 경로
label_zip_dir = r"C:\Users\hotse\Downloads\166.약품식별 인공지능 개발을 위한 경구약제 이미지 데이터\01.데이터\1.Training\라벨링데이터\단일경구약제 5000종"

# 2. 필터 이미지 폴더 경로
filtered_image_dir = r"C:\Users\hotse\Downloads\166.약품식별 인공지능 개발을 위한 경구약제 이미지 데이터\01.데이터\1.Training\원천데이터\단일경구약제 5000종\filtered_images"

# 3. 결과 CSV 파일 경로
output_csv = "labels_existing.csv"
results = []

# 4. 필터 이미지 폴더 내 이미지 파일명 목록 추출
filtered_image_name_set = set(os.listdir(filtered_image_dir))

# 5. 라벨링 ZIP 파일 반복
for i in range(1, 82):
    label_zip_path = os.path.join(label_zip_dir, f"TL_{i}_단일.zip")

    if not os.path.exists(label_zip_path):
        print(f"[!] 라벨링 ZIP 없음: TL_{i}_단일.zip")
        continue

    try:
        with zipfile.ZipFile(label_zip_path, 'r') as lzip:
            for fname in lzip.namelist():
                if fname.endswith(".json"):
                    try:
                        with lzip.open(fname) as f:
                            data = json.load(f)

                            # 'images' 키가 있는 경우 처리
                            if "images" in data:
                                for image_info in data["images"]:
                                    file_name = image_info.get("file_name", "")
                                    label = image_info.get("dl_name", "")

                                    if file_name in filtered_image_name_set:
                                        results.append((file_name, label))
                    except Exception as e:
                        print(f"[!] JSON 파싱 실패 ({fname}): {e}")
    except zipfile.BadZipFile:
        print(f"[!] ZIP 손상됨: {label_zip_path}")
        continue

print(f"총 유효 이미지 수: {len(results)}")

# 6. CSV로 저장
with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label"])
    writer.writerows(results)

print(f"✅ filtered_images 기준 labels_existing.csv 저장 완료")
