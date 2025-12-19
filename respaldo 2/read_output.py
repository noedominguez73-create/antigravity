import sys

input_file = sys.argv[1] if len(sys.argv) > 1 else 'output_users.txt'
output_file = 'output_utf8_' + input_file

content = ""
try:
    with open(input_file, 'r', encoding='utf-16') as f:
        content = f.read()
except Exception:
    with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

with open(output_file, 'w', encoding='utf-8') as f:
    f.write(content)
print(f"Converted to {output_file}")
