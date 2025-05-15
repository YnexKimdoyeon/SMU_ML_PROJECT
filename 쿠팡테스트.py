import undetected_chromedriver
from bs4 import BeautifulSoup
from selenium.webdriver.chrome.options import Options
import requests
import time
import subprocess
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# 크롬 실행 경로
chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

subprocess.Popen([
    chrome_path,
    f'--remote-debugging-port=9229',
])

time.sleep(3)

options = Options()
options.add_experimental_option("debuggerAddress", "127.0.0.1:9229")
print("넘어감")
driver = webdriver.Chrome(options=options)
print("넘어감")
# 4. 테스트 페이지 열기
driver.get("https://www.naver.com")

#
# # 3. Requests 세션에 쿠키 적용
# session = requests.Session()
# for cookie in selenium_cookies:
#     session.cookies.set(cookie['name'], cookie['value'])
#
# # 4. Headers 설정
# headers = {
#     "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
#     "accept-encoding": "gzip, deflate, br, zstd",
#     "accept-language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
#     "cache-control": "max-age=0",
#     "referer": search_url,
#     "sec-ch-ua": '"Chromium";v="136", "Google Chrome";v="136", "Not.A/Brand";v="99"',
#     "sec-ch-ua-mobile": "?0",
#     "sec-ch-ua-platform": '"Windows"',
#     "sec-fetch-dest": "document",
#     "sec-fetch-mode": "navigate",
#     "sec-fetch-site": "same-origin",
#     "sec-fetch-user": "?1",
#     "upgrade-insecure-requests": "1",
#     "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
# }
#
# # 5. 반복적으로 요청 및 파싱
# for i in range(500):
#     response = session.get(search_url, headers=headers)
#     soup = BeautifulSoup(response.text, "html.parser")
#     product_items = soup.select("ul#productList > li")
#
#     print(f"[{i+1}] 상품 개수: {len(product_items)}")
#     if len(product_items) == 0:
#         print("❗ 페이지 HTML 스니펫:")
#         print(response.text[:1000])
#     for idx, item in enumerate(product_items, 1):
#         print(f"{idx}. {item.get_text(strip=True)[:100]}...")
#
#     time.sleep(2)
