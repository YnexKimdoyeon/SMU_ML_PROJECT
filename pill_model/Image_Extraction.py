import os
import zipfile
from collections import defaultdict

# ✅ 설정
base_dir = r"C:\Users\hotse\Downloads\166.약품식별 인공지능 개발을 위한 경구약제 이미지 데이터\01.데이터\1.Training\원천데이터\단일경구약제 5000종"
output_dir = os.path.join(base_dir, "filtered_images")
os.makedirs(output_dir, exist_ok=True)

ALLOWED_EXT = {".jpg", ".png"}  # 확장자 제한
MAX_PER_PILL = 10  # 약제당 최대 추출 수

# ✅ 유효 파일 여부 검사 (조건 최소화)
def is_valid_filename(fname):
    base_name = os.path.basename(fname)
    name, ext = os.path.splitext(base_name)
    if ext.lower() not in ALLOWED_EXT:
        return False

    parts = name.split("_")
    if len(parts) < 8:
        return False  # 기본 구조는 유지

    return True

# ✅ ZIP 반복 처리
for i in range(1, 82):  # 1 ~ 81
    zip_name = f"TS_{i}_단일.zip"
    zip_path = os.path.join(base_dir, zip_name)

    if not os.path.exists(zip_path):
        print(f"[!] {zip_name} 없음. 건너뜀.")
        continue

    print(f"📦 {zip_name} 처리 중...")

    pill_image_count = defaultdict(int)
    valid_image_map = defaultdict(list)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for fname in zip_ref.namelist():
                base_name = os.path.basename(fname)
                if not is_valid_filename(fname):
                    continue

                pill_code = base_name.split("_")[0]
                valid_image_map[pill_code].append(base_name)

                if pill_image_count[pill_code] >= MAX_PER_PILL:
                    continue

                try:
                    out_path = os.path.join(output_dir, base_name)
                    if not os.path.exists(out_path):
                        with zip_ref.open(fname) as src, open(out_path, "wb") as dst:
                            dst.write(src.read())
                        pill_image_count[pill_code] += 1
                except Exception as e:
                    print(f"[!] {fname} 처리 오류: {e}")

        # ✅ 로그 출력
        print(f"🔍 {zip_name} 약제별 추출 현황:")
        for pill_code, images in valid_image_map.items():
            print(f"  - {pill_code}: 조건 만족 {len(images)}장, 추출 {pill_image_count[pill_code]}장")

    except zipfile.BadZipFile:
        print(f"[🚫] {zip_name} → 유효하지 않은 ZIP 파일 (건너뜀)")
        continue

print(f"\n✅ 전체 완료! 추출된 이미지 저장 위치:\n→ {output_dir}")
