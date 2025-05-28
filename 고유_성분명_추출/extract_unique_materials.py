import zipfile
import os
import json
import csv
import re

# 문자열 정제 함수
def clean_material_name(text):
    # 쉼표, %, 숫자, 단위 제거
    text = re.sub(r'[\d.,%㎎㎍]', '', text)
    return text.strip()

zip_dir = os.path.dirname(os.path.abspath(__file__))
output_csv = os.path.join(zip_dir, "unique_dl_material_en.csv")

seen_materials = set()
results = []

for i in range(1, 82):
    zip_name = f"TL_{i}_단일.zip"
    zip_path = os.path.join(zip_dir, zip_name)

    if not os.path.exists(zip_path):
        print(f"[!] {zip_name} not found. Skipping.")
        continue

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for fname in zip_ref.namelist():
            if fname.endswith(".json"):
                try:
                    with zip_ref.open(fname) as f:
                        data = json.load(f)

                        if isinstance(data, dict) and "images" in data:
                            for img in data["images"]:
                                material_str = img.get("dl_material_en")
                                if material_str:
                                    components = [clean_material_name(c) for c in material_str.split("|")]
                                    for comp in components:
                                        if comp and comp not in seen_materials:
                                            seen_materials.add(comp)
                                            results.append({
                                                "dl_material_en": comp
                                            })
                except Exception as e:
                    print(f"[!] Error parsing {fname} in {zip_name}: {e}")

# CSV로 저장 (dl_material_en만)
with open(output_csv, "w", encoding="utf-8", newline='') as csvfile:
    fieldnames = ["dl_material_en"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"✅ 완료: {len(results)}개의 정제된 고유 성분명이 {output_csv} 에 저장되었습니다.")
