
import os

root_dir = r'c:\asesoriaimss.io'
print(f"Listing directories in {root_dir}...")

for name in os.listdir(root_dir):
    if 'respaldo' in name:
        print(f"'{name}' - Is Dir: {os.path.isdir(os.path.join(root_dir, name))}")
