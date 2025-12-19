import shutil
import os
import datetime

def backup_project():
    source_dir = os.getcwd()
    backup_dir = os.path.join(source_dir, 'respaldo creditos')
    
    if os.path.exists(backup_dir):
        print(f"Backup directory {backup_dir} already exists. Removing it...")
        shutil.rmtree(backup_dir)
        
    print(f"Creating backup at {backup_dir}...")
    os.makedirs(backup_dir)
    
    # Files/Dirs to backup
    items_to_backup = [
        'app',
        'run.py',
        'requirements.txt',
        '.env',
        'reingest_all.py',
        'debug_rag.py'
    ]
    
    for item in items_to_backup:
        s = os.path.join(source_dir, item)
        d = os.path.join(backup_dir, item)
        
        if os.path.exists(s):
            if os.path.isdir(s):
                print(f"Copying directory: {item}")
                shutil.copytree(s, d)
            else:
                print(f"Copying file: {item}")
                shutil.copy2(s, d)
        else:
            print(f"Skipping {item} (not found)")
            
    print("Backup completed successfully!")

if __name__ == "__main__":
    backup_project()
