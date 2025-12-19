import shutil
import os
import datetime

def create_backup():
    source_dir = r"c:\asesoriaimss.io"
    # Create valid timestamp for folder name if needed, but user asked for "respaldo" implicitly implied next one
    # I will use "respaldo 2" as per the pattern of "respaldo 1"
    backup_dir_name = "respaldo 2" 
    destination_dir = os.path.join(source_dir, backup_dir_name)

    print(f"Starting backup from {source_dir} to {destination_dir}...")

    if os.path.exists(destination_dir):
        print(f"Backup directory {destination_dir} already exists. Removing it to create a fresh backup...")
        shutil.rmtree(destination_dir)

    def ignore_patterns(path, names):
        # Ignore existing backups, git, venv, and cache
        ignore_list = ['.git', '__pycache__', 'venv', 'node_modules', 'respaldo 1', 'respaldo 2', 'respaldo 3', 'respaldo*']
        return ignore_list

    try:
        shutil.copytree(source_dir, destination_dir, ignore=ignore_patterns)
        print(f"Backup completed successfully to {destination_dir}")
    except Exception as e:
        print(f"Backup failed: {e}")

if __name__ == "__main__":
    create_backup()
