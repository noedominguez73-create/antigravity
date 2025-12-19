
import shutil
import os

source = r'c:\asesoriaimss.io\respaldo 1'
destination = r'c:\asesoriaimss.io'

print(f"Restoring from {source} to {destination}...")

for item in os.listdir(source):
    s = os.path.join(source, item)
    d = os.path.join(destination, item)
    
    if os.path.isdir(s):
        print(f"Copying directory {item}...")
        try:
            shutil.copytree(s, d, dirs_exist_ok=True)
        except Exception as e:
            print(f"Error copying directory {item}: {e}")
    else:
        print(f"Copying file {item}...")
        try:
            shutil.copy2(s, d)
        except Exception as e:
            print(f"Error copying file {item}: {e}")

print("Restoration complete.")
