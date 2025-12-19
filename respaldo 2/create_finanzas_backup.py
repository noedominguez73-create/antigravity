import os
import shutil
from datetime import datetime

def create_backup():
    # Source directories
    base_dir = r"c:\asesoriaimss.io"
    templates_dir = os.path.join(base_dir, "app", "templates")
    static_js_dir = os.path.join(base_dir, "app", "static", "js")

    # Generate timestamped backup folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = os.path.join(base_dir, "backups")
    backup_dir = os.path.join(backup_root, f"backup_mis_finanzas_{timestamp}")

    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    # Files to backup
    files_to_backup = [
        # Templates
        os.path.join(templates_dir, "mis_finanzas_dashboard.html"),
        os.path.join(templates_dir, "mis_finanzas_facturas.html"),
        os.path.join(templates_dir, "mis_finanzas_ingresos.html"),
        os.path.join(templates_dir, "mis_finanzas_pagos.html"),
        os.path.join(templates_dir, "mis_finanzas_pendientes.html"),
        os.path.join(templates_dir, "mis_finanzas_reportes.html"),
        # JS
        os.path.join(static_js_dir, "finance-core.js"),
        os.path.join(static_js_dir, "finance-ai.js"),
        # Nav component if exists
        os.path.join(templates_dir, "components", "finance_nav.html")
    ]

    print(f"Creating backup at: {backup_dir}")

    for file_path in files_to_backup:
        if os.path.exists(file_path):
            filename = os.path.basename(file_path)
            dest_path = os.path.join(backup_dir, filename)
            shutil.copy2(file_path, dest_path)
            print(f"Backed up: {filename}")
        else:
            print(f"Warning: File not found {file_path}")

    print("Backup completed successfully.")

if __name__ == "__main__":
    create_backup()
