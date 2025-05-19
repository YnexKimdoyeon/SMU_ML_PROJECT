import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time

# 설정
MAX_WORKERS = 50
SAVE_EVERY = 1000
BATCH_SIZE = 5000
CSV_FILENAME = "drug_interactions_partial.csv"
COMPLETE_LOG = "completed_pairs.txt"

# 정규식으로 코드/이름 추출
def return_match(text):
    match = re.search(r'selectAdd\("([^"]+)","([^"]+)","([^"]+)"\)', text)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return None, None, None

# 요청 헤더
url = "https://www.health.kr/interaction/drug.asp"
headers = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "content-type": "application/x-www-form-urlencoded",
    "referer": "https://www.health.kr/interaction/drug.asp"
}

# 데이터 로딩
with open("file.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f.readlines()]
drug_list = [return_match(line) for line in lines if return_match(line)[0]]

# 완료된 조합 로딩
completed = set()
if os.path.exists(COMPLETE_LOG):
    with open(COMPLETE_LOG, "r", encoding="utf-8") as f:
        completed = set(line.strip() for line in f.readlines())

# 상호작용 확인 함수
def check_interaction(drug1, drug2, retries=2):
    one_index, one_code, one_name = drug1
    two_index, two_code, two_name = drug2
    key = f"{one_name}|{two_name}"

    if key in completed:
        return None

    data = [
        ("inits", "2"),
        ("selectCount", "2"),
        ("interaction_search_word", one_name),
        ("numid", "11"),
        ("sunb_name", one_name),
        ("ingd_code", one_code),
        ("numids", one_index),
        ("sunb_name", two_name),
        ("ingd_code", two_code),
        ("numids", two_index),
    ]

    for attempt in range(retries):
        try:
            response = requests.post(url, data=data, headers=headers, timeout=10)
            if response.status_code != 200:
                print(f"❌ 응답 오류 {response.status_code} - {one_name} + {two_name}")
                return None

            soup = BeautifulSoup(response.text, "html.parser")
            info_cells = soup.select("td.info ul li")

            if len(info_cells) >= 3:
                print(f"✅ {one_name} + {two_name} → 상호작용 발견!")
                return {
                    "약물1": one_name,
                    "약물1코드": one_code,
                    "약물2": two_name,
                    "약물2코드": two_code,
                    "임상효과": info_cells[-3].get_text(strip=True),
                    "기전": info_cells[-2].get_text(strip=True),
                    "처치": info_cells[-1].get_text(strip=True),
                    "key": key
                }
            else:
                print(f"❌ {one_name} + {two_name} → 상호작용 없음")
                return {
                    "약물1": one_name,
                    "약물1코드": one_code,
                    "약물2": two_name,
                    "약물2코드": two_code,
                    "임상효과": "X",
                    "기전": "X",
                    "처치": "X",
                    "key": key
                }
        except Exception as e:
            print(f"⚠️ 예외 발생: {one_name} + {two_name} | {e}")
            time.sleep(1)
    return {
        "약물1": one_name,
        "약물1코드": one_code,
        "약물2": two_name,
        "약물2코드": two_code,
        "임상효과": "X",
        "기전": "X",
        "처치": "X",
        "key": key
    }

# CSV 저장 함수
def save_results(batch_results):
    df = pd.DataFrame(batch_results)
    if os.path.exists(CSV_FILENAME):
        old = pd.read_csv(CSV_FILENAME, encoding="utf-8-sig")
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(CSV_FILENAME, index=False, encoding="utf-8-sig")

    with open(COMPLETE_LOG, "a", encoding="utf-8") as f:
        for r in batch_results:
            f.write(r["key"] + "\n")

# 조합 만들기
all_pairs = [
    (drug_list[i], drug_list[j])
    for i in range(len(drug_list))
    for j in range(i + 1, len(drug_list))
    if f"{drug_list[i][2]}|{drug_list[j][2]}" not in completed
]

print(f"🔍 총 처리 대상 조합 수: {len(all_pairs)}")

# 병렬 실행 및 저장 루프
results = []
count = 0
for batch_start in range(0, len(all_pairs), BATCH_SIZE):
    batch = all_pairs[batch_start:batch_start + BATCH_SIZE]
    print(f"\n🚀 {batch_start + 1} ~ {batch_start + len(batch)} 조합 처리 시작")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(check_interaction, a, b): (a, b) for a, b in batch}

        for future in as_completed(futures):
            a, b = futures[future]
            count += 1
            try:
                res = future.result()
                if res:
                    results.append(res)
            except Exception as e:
                print(f"⚠️ 처리 실패: {a[2]} + {b[2]} | {e}")

            if len(results) >= SAVE_EVERY:
                print(f"\n💾 {len(results)}건 중간 저장 중...")
                save_results(results)
                results.clear()

# 마지막 저장
if results:
    print(f"\n📦 마지막 {len(results)}건 저장")
    save_results(results)

print("\n✅ 전체 완료 및 저장됨")
