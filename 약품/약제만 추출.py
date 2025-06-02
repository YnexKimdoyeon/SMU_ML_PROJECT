import re


with open ("result.txt", "r", encoding="utf-8") as f:
    data = f.read()


ingredients = re.findall(r'\"([^"]+)\"\)\s*$', data, re.MULTILINE)

for i in ingredients:
    with open ("약학정보원 데이터.txt", "a", encoding="utf-8") as f:
        f.write(i+"\n")