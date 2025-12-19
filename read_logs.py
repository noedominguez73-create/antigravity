import os

log_path = 'logs/asesoriaimss.log'
if os.path.exists(log_path):
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
        print("Last 500 log lines:")
        for line in lines[-500:]:
            print(line.strip())
else:
    print("Log file not found.")
