import re

with open('index.html', 'r') as f:
    html = f.read()

with open('app.js', 'r') as f:
    js = f.read()

ids_in_js = re.findall(r'getElementById\([\'"]([^\'"]+)[\'"]\)', js)
ids_in_js = set(ids_in_js)

missing = []
for i in ids_in_js:
    if f'id="{i}"' not in html and f"id='{i}'" not in html:
        missing.append(i)

print("Missing IDs:", missing)
