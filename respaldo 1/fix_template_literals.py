
import re

file_path = r'c:\asesoriaimss.io\app\templates\mis_finanzas.html'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: Replace "$ {" with "${" (handling potential newlines)
# The pattern seems to be "$ {" followed by optional whitespace/newlines
# We want to collapse "$ {\s*VAR\s*}" to "${VAR}"

# Regex to find $ { ... } and collapse it
# Matches $ { followed by whitespace, then content, then whitespace, then }
# We need to be careful not to match too much.
# The content inside usually doesn't contain }
pattern = r'\$\s*\{\s*([^}]+?)\s*\}'

def replacer(match):
    return f"${{{match.group(1).strip()}}}"

new_content = re.sub(pattern, replacer, content)

# Also fix the specific case where it might be split across lines like:
# `$ {
#    variable
# }`
# The above regex should handle it if `\s*` matches newlines.

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Fixed template literals.")
