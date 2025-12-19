"""
Script to update .env file to use SQLite
"""
import os

env_path = '.env'

# Read current .env
with open(env_path, 'r') as f:
    lines = f.readlines()

# Update DATABASE_URL line
new_lines = []
for line in lines:
    if line.startswith('DATABASE_URL='):
        new_lines.append('DATABASE_URL=sqlite:///asesoriaimss.db\n')
        print('✓ Updated DATABASE_URL to use SQLite')
    else:
        new_lines.append(line)

# Write back to .env
with open(env_path, 'w') as f:
    f.writelines(new_lines)

print('✓ .env file updated successfully')
