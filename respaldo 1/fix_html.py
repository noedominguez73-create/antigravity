import sys

# Read the file
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    content = f.read()

# Replace escaped unicode
content = content.replace('\\u003c', '<').replace('\\u003e', '>')

# Write back
with open(sys.argv[1], 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Fixed {sys.argv[1]}")
