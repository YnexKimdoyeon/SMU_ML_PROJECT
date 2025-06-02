import zipfile
import os
import json

zip_dir = os.path.dirname(os.path.abspath(__file__))
output_txt = os.path.join(zip_dir, "unique_dl_material_en.txt")

seen_materials = set()

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
                                    components = [c.strip() for c in material_str.split("|")]
                                    for comp in components:
                                        if comp:
                                            seen_materials.add(comp)
                except Exception as e:
                    print(f"[!] Error parsing {fname} in {zip_name}: {e}")

# TXT로 저장
with open(output_txt, "w", encoding="utf-8") as f:
    for comp in sorted(seen_materials):
        f.write(comp + "\n")

print(f"✅ 완료: {len(seen_materials)}개의 고유 성분명이 {output_txt} 에 저장되었습니다.")
