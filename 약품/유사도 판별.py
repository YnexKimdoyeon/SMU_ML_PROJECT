import difflib
import pandas as pd

# 파일 경로
unmatched_path = "노매칭데이터.txt"
source_path = "약학정보원 데이터.txt"

# 노매칭 데이터 로드
with open(unmatched_path, "r", encoding="utf-8") as f:
    unmatched_list = [line.strip() for line in f if line.strip()]

# 약학정보원 데이터 로드
with open(source_path, "r", encoding="utf-8") as f:
    source_list = [line.strip() for line in f if line.strip()]

# 가장 유사한 항목 찾기
replacement_result = []
for item in unmatched_list:
    best_match = difflib.get_close_matches(item, source_list, n=1, cutoff=0.6)
    replacement_result.append((item, best_match[0] if best_match else "❌ 매칭 실패"))

# 데이터프레임으로 정리
df_replaced = pd.DataFrame(replacement_result, columns=["원본 성분명", "대체된 성분명"])

# 엑셀로 저장
output_path = "노매칭_대체_결과.xlsx"
df_replaced.to_excel(output_path, index=False, engine='openpyxl')
print(f"✅ 결과가 '{output_path}'에 저장되었습니다.")
