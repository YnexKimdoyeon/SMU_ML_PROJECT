import time

from selenium import webdriver
from selenium.webdriver.common.by import By

driver = webdriver.Chrome()
driver.get("https://www.health.kr/interaction/drug.asp")
driver.implicitly_wait(5)

alist = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
c = []
for j in alist:
    print(j)
    driver.find_element(By.ID,'interaction_search_word').send_keys(j)
    driver.find_element(By.CSS_SELECTOR,'button.btn02').click()
    time.sleep(1)
    file = driver.find_elements(By.CSS_SELECTOR,"article.schBox li")
    for i in file:
        try:
            with open ("file.txt","a",encoding="UTF-8") as filee:
                if i.find_element(By.CSS_SELECTOR,'a').get_attribute('onclick')+"\n" not in c:
                    c.append(i.find_element(By.CSS_SELECTOR,'a').get_attribute('onclick')+"\n")
                    filee.write(i.find_element(By.CSS_SELECTOR,'a').get_attribute('onclick')+"\n")
        except:
            pass
    driver.refresh()