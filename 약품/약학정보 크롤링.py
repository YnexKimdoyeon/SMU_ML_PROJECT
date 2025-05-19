import time

from openpyxl.workbook import Workbook
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

driver = webdriver.Chrome()
driver.get("https://www.health.kr/interaction/drug.asp")
driver.implicitly_wait(5)
start = int(input("시작값 : "))
end = int(input("종료값 : "))
with open ("file.txt", "r",encoding="UTF-8") as f:
    alist = f.readlines()
for j in range(start, end):
    print(j)
    try:
        driver.execute_script(alist[j])
    except:
        print("error")
    time.sleep(0.1)
time.sleep(1000)
driver.execute_script('searchProduct();')
print("추출시작")

html = driver.page_source
soup = BeautifulSoup(html, "html.parser")
table = soup.select_one("table#result_drug")

wb = Workbook()
ws = wb.active
ws.title = "결과 테이블"

for tr in table.select("tr"):
    cells = tr.find_all(["th", "td"])
    row_data = [cell.get_text(strip=True) for cell in cells]
    ws.append(row_data)

wb.save("result_drug_table.xlsx")