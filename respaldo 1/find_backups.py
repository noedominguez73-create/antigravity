
import os
import datetime

root_dir = r'c:\asesoriaimss.io'
target_date = datetime.date(2025, 12, 4)

print(f"Searching for directories modified on {target_date} in {root_dir}...")

found = []

for root, dirs, files in os.walk(root_dir):
    # Skip venv and .git to save time
    if '.venv' in dirs:
        dirs.remove('.venv')
    if '.git' in dirs:
        dirs.remove('.git')
    if 'venv_test' in dirs:
        dirs.remove('venv_test')
        
    for d in dirs:
        full_path = os.path.join(root, d)
        try:
            mtime = os.path.getmtime(full_path)
            mdate = datetime.date.fromtimestamp(mtime)
            if mdate == target_date:
                found.append(full_path)
        except OSError:
            pass

for p in found:
    print(p)

if not found:
    print("No directories found from Dec 4th.")
