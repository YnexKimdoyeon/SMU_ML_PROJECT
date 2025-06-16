import chardet
import pandas as pd
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

# 파일명 직접 지정 (같은 폴더에 있는 경우)
interaction_file = "drug_interactions_partial.csv"       # 원본 성분명, 대체된 성분명 포함
main_data_file = r"C:\Users\Doyeon\Desktop\기프 프로젝트\약품\노매칭_대체_결과.xlsx.csv"     # 약물1, 약물2 포함

main_encoding = detect_encoding(main_data_file)
interaction_encoding = detect_encoding(interaction_file)

main_df = pd.read_csv(main_data_file, encoding=main_encoding)
interaction_df = pd.read_csv(interaction_file, encoding=interaction_encoding)
replacement_dict = dict(zip(main_df['대체된 성분명'], main_df['원본 성분명']))

# 치환 적용
interaction_df['약물1'] = interaction_df['약물1'].replace(replacement_dict)
interaction_df['약물2'] = interaction_df['약물2'].replace(replacement_dict)

# 결과 저장 (선택사항)
interaction_df.to_csv("replaced_interaction_data.csv", index=False, encoding="utf-8-sig")


# 결과 출력
print(interaction_df.head())
