#
#
# with open ("약학정보원 데이터.txt","r",encoding="utf-8") as f:
#     약학데이터 = f.readlines()
#
#
#
# with open ("json데이터.txt","r",encoding="utf-8") as f:
#     JSON데이터 = f.readlines()
#
#
# result = []
# noresult = []
# for i in JSON데이터:
#     if i in 약학데이터:
#         result.append(i)
#     else:
#         noresult.append(i)
#
#
# with open ("매칭데이터.txt","a",encoding="utf-8") as f:
#     for i in result:
#         f.write(i)
#
# with open ("노매칭데이터.txt","a",encoding="utf-8") as f:
#     for i in noresult:
#         f.write(i)


# with open ("매칭데이터.txt",'r',encoding="utf-8") as f:
#     li = f.readlines()
#
# with open ("매칭결과데이터.txt",'a',encoding="utf-8") as f:
#     for i in li:
#         if i.strip() != "":
#             f.write(i.rstrip())
#             f.write("\n")

#
# 파일 읽기
with open("결과.txt", 'r', encoding="utf-8") as f:
    li = [line.strip() for line in f if line.strip()]
with open("file.txt", 'r', encoding="utf-8") as f:
    li2 = [line.strip() for line in f if line.strip()]
matched_lines = []

for line2 in li2:
    if any(line1+'")' in line2 for line1 in li):
        matched_lines.append(line2)
with open("매칭결과.txt", 'w', encoding="utf-8") as f:
    for line in matched_lines:
        f.write(line + "\n")
print(f"✅ 총 {len(matched_lines)}개의 줄이 매칭되어 '매칭결과.txt'에 저장되었습니다.")
