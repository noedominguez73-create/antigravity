
import re

file_path = r'c:\asesoriaimss.io\app\templates\index.html'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Target pattern for the description
desc_pattern = r'<p class="text-gray-500 text-sm mb-4 flex-grow">Planifica tus outfits para la semana según el clima\s+y eventos.</p>'
desc_replacement = '<p class="text-gray-500 text-sm mb-4 flex-grow">Administra tus ingresos, gastos y facturas en un solo lugar.</p>'

# Target pattern for the button
btn_pattern = r'<button disabled\s+class="block w-full py-2 text-center bg-gray-100 text-gray-400 rounded-lg font-semibold cursor-not-allowed text-sm">Próximamente</button>'
btn_replacement = '<a href="/mis-finanzas"\n                        class="block w-full py-2 text-center bg-green-600 text-white rounded-lg font-semibold hover:bg-green-700 transition text-sm">Gestionar</a>'

# Perform replacements
new_content = re.sub(desc_pattern, desc_replacement, content)
new_content = re.sub(btn_pattern, btn_replacement, new_content)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Replaced content in index.html")
