import re


def return_match(text):
    match = re.search(r'selectAdd\("([^"]+)","([^"]+)","([^"]+)"\)', text)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return None, None, None

result = []
with open ("backup.txt","r",encoding="utf-8") as f:
    f = f.readlines()

for i in f:
    a = return_match(i)
    if a[1] not in result:
        result.append(a[1])
        with open("result.txt", "a", encoding="utf-8") as f:
            f.write(i)

