
import os
import shutil
import datetime

def backup_files():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(os.getcwd(), 'backups', f'mirror_admin_complete_{timestamp}')
    
    os.makedirs(backup_dir, exist_ok=True)
    
    files_to_backup = [
        'app/routes/mirror_api.py',
        'app/templates/admin_mirror.html',
        'app/templates/mirror.html',
        'app/models.py'
    ]
    
    print(f"Creating backup at: {backup_dir}")
    
    for relative_path in files_to_backup:
        src = os.path.join(os.getcwd(), relative_path)
        dst = os.path.join(backup_dir, os.path.basename(relative_path))
        
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Backed up: {relative_path}")
        else:
            print(f"Warning: Source file not found: {relative_path}")
            
    print("Backup completed successfully.")

if __name__ == "__main__":
    backup_files()
