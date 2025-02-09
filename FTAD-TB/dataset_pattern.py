import re

# 读取原始文件内容
with open("./dataset.json", "r", encoding="utf-8") as f:
    text = f.read()

# 正则表达式匹配 "width": 后面跟任意数字，并匹配 "height": 后面跟任意数字
pattern = r'\s*"width":\s*\d+,\s*\n\s*"height":\s*\d+,?'

# 执行正则替换，去掉匹配内容
new_text = re.sub(pattern, "", text)

# 将替换后的内容写入新文件
with open("dataset_new.json", "w", encoding="utf-8") as f:
    f.write(new_text)

