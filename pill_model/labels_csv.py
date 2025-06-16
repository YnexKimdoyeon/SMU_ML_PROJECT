import os
import zipfile
import json
import csv

# ZIP 파일이 들어있는 디렉터리
zip_dir = r"C:\Users\hotse\Downloads\166.약품식별 인공지능 개발을 위한 경구약제 이미지 데이터\01.데이터\1.Training\라벨링데이터\단일경구약제 5000종"
output_csv = "labels.csv"

results = []

for i in range(1, 82):  # TL_1 ~ TL_81
    zip_name = f"TL_{i}_단일.zip"
    zip_path = os.path.join(zip_dir, zip_name)

    if not os.path.exists(zip_path):
        print(f"[!] {zip_name} 없음. 스킵합니다.")
        continue

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for fname in zip_ref.namelist():
            if fname.endswith(".json"):
                try:
                    with zip_ref.open(fname) as f:
                        data = json.load(f)
                        image_info = data.get("images", [{}])[0]
                        file_name = image_info.get("file_name", "")
                        label = image_info.get("dl_name", "")

                        if file_name and label:
                            results.append((file_name, label))
                except Exception as e:
                    print(f"[!] {fname} 읽는 중 오류 발생: {e}")

# CSV로 저장
with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label"])
    writer.writerows(results)

print(f"✅ 완료: 총 {len(results)}개의 레이블이 labels.csv에 저장되었습니다.")
