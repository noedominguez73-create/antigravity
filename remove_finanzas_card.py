
import re

file_path = r'c:\asesoriaimss.io\app\templates\perfil-usuario.html'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Pattern to match the Mis Finanzas card block
# We match from <!-- Mis Finanzas Card --> to the closing </a> tag
pattern = r'<!-- Mis Finanzas Card -->\s*<a href="/mis-finanzas".*?</a>'

# Replace with empty string
new_content = re.sub(pattern, '', content, flags=re.DOTALL)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Removed Mis Finanzas card from perfil-usuario.html")
