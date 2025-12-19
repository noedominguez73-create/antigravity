import shutil
import os
from datetime import datetime

def create_backup():
    # Setup paths
    source_dir = os.getcwd()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_name = f"backup_mirror_refactor_{timestamp}"
    backup_dir = os.path.join(source_dir, 'backups', backup_name)
    
    # Create backups folder if it doesn't exist
    if not os.path.exists(os.path.join(source_dir, 'backups')):
        os.makedirs(os.path.join(source_dir, 'backups'))
        
    print(f"Creating backup at: {backup_dir}")
    os.makedirs(backup_dir)
    
    # Items to backup
    items_to_backup = [
        'app',
        'run.py',
        'requirements.txt',
        '.env',
        'config.py' # If exists
    ]
    
    # Perform Copy
    for item in items_to_backup:
        s = os.path.join(source_dir, item)
        d = os.path.join(backup_dir, item)
        
        if os.path.exists(s):
            if os.path.isdir(s):
                # Ignore __pycache__ and 'uploads' to save space/time if needed, 
                # but for now full backup involves uploads too? 
                # Maybe skip static/uploads if huge? 
                # User said "respalda", usually implies safely code. 
                # Let's exclude __pycache__.
                shutil.copytree(s, d, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo'))
                print(f"✔ Copied directory: {item}")
            else:
                shutil.copy2(s, d)
                print(f"✔ Copied file: {item}")
        else:
            print(f"⚠ Skipped missing: {item}")
            
    print(f"\nSUCCESS: Backup created at 'backups/{backup_name}'")

if __name__ == "__main__":
    create_backup()
