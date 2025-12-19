import shutil
import os
import sys

def restore_backup():
    backup_dir = r"c:\asesoriaimss.io\respaldo 2"
    target_dir = r"c:\asesoriaimss.io"

    if not os.path.exists(backup_dir):
        print(f"Error: Backup directory {backup_dir} does not exist!")
        sys.exit(1)

    print(f"Restoring from {backup_dir} to {target_dir}...")
    print("WARNING: This will overwrite files in the target directory.")

    # We want to copy everything FROM backup TO target
    # We must be careful not to delete the 'respaldo *' folders in the target if they exist
    # shutil.copytree with dirs_exist_ok=True works for merging/overwriting files

    try:
        # Walk through backup dir and copy files to target
        for root, dirs, files in os.walk(backup_dir):
            # Calculate relative path
            rel_path = os.path.relpath(root, backup_dir)
            target_root = os.path.join(target_dir, rel_path)

            if not os.path.exists(target_root):
                os.makedirs(target_root)

            for file in files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(target_root, file)
                try:
                    shutil.copy2(src_file, dst_file)
                    # print(f"Restored: {dst_file}") 
                except Exception as e:
                    print(f"Failed to copy {src_file}: {e}")
        
        print("Restore completed successfully.")

    except Exception as e:
        print(f"Critical Error during restore: {e}")
        sys.exit(1)

if __name__ == "__main__":
    restore_backup()
