import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# 정규식으로 코드, 이름 추출
def return_match(text):
    match = re.search(r'selectAdd\("([^"]+)","([^"]+)","([^"]+)"\)', text)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return None, None, None

# 요청 헤더
url = "https://www.health.kr/interaction/drug.asp"
headers = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "accept-encoding": "gzip, deflate, br, zstd",
    "accept-language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "cache-control": "max-age=0",
    "connection": "keep-alive",
    "content-type": "application/x-www-form-urlencoded",
    "origin": "https://www.health.kr",
    "referer": "https://www.health.kr/interaction/drug.asp",
    "sec-ch-ua": '"Chromium";v="136", "Google Chrome";v="136", "Not.A/Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "same-origin",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
    "host": "www.health.kr",
}

# 파일 로딩
with open("file.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f.readlines()]

drug_list = []
for line in lines:
    idx, code, name = return_match(line)
    if idx:
        drug_list.append((idx, code, name))


def check_interaction(drug1, drug2):
    one_index, one_code, one_name = drug1
    two_index, two_code, two_name = drug2

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

    try:
        response = requests.post(url, data=data, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"❌ 응답코드 {response.status_code} - {one_name} + {two_name}")
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        info_cells = soup.select("td.info ul li")

        print(f"🔍 {one_name} + {two_name} → info {len(info_cells)}개")

        if len(info_cells) >= 3:
            return {
                "약물1": one_name,
                "약물1코드": one_code,
                "약물2": two_name,
                "약물2코드": two_code,
                "임상효과": info_cells[-3].get_text(strip=True),
                "기전": info_cells[-2].get_text(strip=True),
                "처치": info_cells[-1].get_text(strip=True)
            }
        else:
            return {
                "약물1": one_name,
                "약물1코드": one_code,
                "약물2": two_name,
                "약물2코드": two_code,
                "임상효과": "X",
                "기전": "X",
                "처치": "X"
            }
    except Exception as e:
        print(f"❗예외 발생: {one_name} + {two_name} | {e}")
        return {
            "약물1": one_name,
            "약물1코드": one_code,
            "약물2": two_name,
            "약물2코드": two_code,
            "임상효과": "X",
            "기전": "X",
            "처치": "X"
        }

# 병렬 실행
results = []
futures = []
with ThreadPoolExecutor(max_workers=10) as executor:
    for i in range(len(drug_list)):
        for j in range(i + 1, len(drug_list)):
            futures.append(executor.submit(check_interaction, drug_list[i], drug_list[j]))

    for future in as_completed(futures):
        try:
            result = future.result()
            if result:
                results.append(result)
        except Exception as e:
            print(f"⚠️ 예외 무시됨: {e}")
# 엑셀 저장
df = pd.DataFrame(results)
df.to_excel("drug_interactions_threaded_100.xlsx", index=False, engine="openpyxl")
print("✅ 병렬 엑셀 저장 완료: drug_interactions_threaded_100.xlsx")
