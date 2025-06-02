import os
import zipfile

# 1. ZIP 파일 경로 (TS_81_단일.zip)
zip_path = r"C:\Users\hotse\Downloads\166.약품식별 인공지능 개발을 위한 경구약제 이미지 데이터\01.데이터\1.Training\원천데이터\단일경구약제 5000종\TS_81_단일.zip"

# 2. 추출된 이미지 저장 폴더
output_dir = os.path.join(os.path.dirname(zip_path), "filtered_images")
os.makedirs(output_dir, exist_ok=True)

# 3. 조건 설정
TARGET_BG_IDX = "2"             # 배경색 index 고정
TARGET_LIGHT = ("60", "160")    # 조도 조건
ALLOWED_ROT_IDX = {"0", "1", "2", "3"}  # 회전 인덱스

# 4. 압축 파일 열기
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    for fname in zip_ref.namelist():
        if not (fname.endswith(".png") or fname.endswith(".jpg")):
            continue

        try:
            # ZIP 내부의 경로: ex) K-038890/K-038890_0_2_1_0_60_160_200.png
            base_name = os.path.basename(fname)
            parts = base_name.split("_")

            if len(parts) < 8:
                continue

            bg_idx = parts[2]
            light_la = parts[5]
            light_lo = parts[6]
            rot_idx = parts[4]

            if (
                bg_idx == TARGET_BG_IDX and
                (light_la, light_lo) == TARGET_LIGHT and
                rot_idx in ALLOWED_ROT_IDX
            ):
                out_path = os.path.join(output_dir, base_name)
                if not os.path.exists(out_path):
                    with zip_ref.open(fname) as src, open(out_path, "wb") as dst:
                        dst.write(src.read())

        except Exception as e:
            print(f"[!] {fname} 처리 오류: {e}")

print(f"✅ 조건을 만족하는 이미지 추출 완료 → {output_dir}")
