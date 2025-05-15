import time

from openpyxl.workbook import Workbook
from selenium import webdriver
from selenium.webdriver.common.by import By

driver = webdriver.Chrome()
driver.get("https://www.health.kr/interaction/drug.asp")
driver.implicitly_wait(5)
with open ("file.txt", "r",encoding="UTF-8") as f:
    alist = f.readlines()
for Number,j in enumerate(alist):
    print(j)
    try:
        driver.execute_script(j)
    except:
        print("error")
        input()
    time.sleep(0.1)
    if Number == 1000:
        print("브렉")
        break

input()
rows = driver.find_elements(By.CSS_SELECTOR, "table#result_drug tr")

# 3. 워크북 생성
wb = Workbook()
ws = wb.active
ws.title = "결과 테이블"

# 4. tr > td 값 추출해서 엑셀에 저장
for row in rows:
    cells = row.find_elements(By.TAG_NAME, "th")  # 제목 셀
    if not cells:
        cells = row.find_elements(By.TAG_NAME, "td")  # 일반 셀
    row_data = [cell.text.strip() for cell in cells]
    ws.append(row_data)

# 5. 엑셀 저장
wb.save("result_drug_table.xlsx")