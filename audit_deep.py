"""
Deep Audit Script for AsesoriaIMSS.io
Covers Security, Error Handling, and Database Schema
"""

import os
import re
import sqlite3
import ast
from pathlib import Path

class DeepAuditor:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.issues = []
        
    def add_issue(self, category, severity, file, message):
        self.issues.append({
            'category': category,
            'severity': severity,
            'file': str(file),
            'message': message
        })

    def check_secrets(self, file_path, content):
        """Check for hardcoded secrets"""
        patterns = {
            'api_key': r'(?i)(api_key|apikey|secret|token)\s*=\s*[\'"][a-zA-Z0-9_\-]{20,}[\'"]',
            'password': r'(?i)(password|passwd|pwd)\s*=\s*[\'"][^\'"]{5,}[\'"]',
        }
        
        # Ignore example/test files
        if 'test' in str(file_path) or 'example' in str(file_path):
            return

        for name, pattern in patterns.items():
            matches = re.finditer(pattern, content)
            for match in matches:
                # Filter out os.getenv calls which might look like assignments if not careful, 
                # but regex above expects quotes, so os.getenv('KEY') is safe, but var = "literal" is caught.
                # Also filter out common placeholders
                val = match.group(0)
                if 'os.getenv' not in val and 'environ.get' not in val and 'placeholder' not in val.lower():
                     self.add_issue('SECURITY', 'HIGH', file_path, f"Potential hardcoded {name}: {val[:20]}...")

    def check_error_handling(self, file_path, content):
        """Check try-except blocks and logging"""
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ExceptHandler):
                    # Check for bare except or Exception
                    if node.type is None:
                        self.add_issue('ERROR_HANDLING', 'MEDIUM', file_path, f"Line {node.lineno}: Bare 'except:' found. Use specific exceptions.")
                    elif isinstance(node.type, ast.Name) and node.type.id == 'Exception':
                        # Check if it logs
                        has_log = False
                        for child in ast.walk(node):
                            if isinstance(child, ast.Call):
                                if isinstance(child.func, ast.Attribute) and 'log' in child.func.attr:
                                    has_log = True
                                elif isinstance(child.func, ast.Name) and 'print' in child.func.id:
                                    self.add_issue('ERROR_HANDLING', 'LOW', file_path, f"Line {node.lineno}: Using 'print' instead of logger in exception handler.")
                        
                        if not has_log:
                             self.add_issue('ERROR_HANDLING', 'MEDIUM', file_path, f"Line {node.lineno}: Generic 'except Exception' without logging.")

        except SyntaxError:
            pass

    def check_database_schema(self):
        """Inspect SQLite database schema"""
        db_path = self.project_root / 'instance' / 'asesoriaimss.db'
        if not db_path.exists():
            # Try root if not in instance
            db_path = self.project_root / 'asesoriaimss.db'
            
        if not db_path.exists():
            self.add_issue('DATABASE', 'INFO', 'DB', "Database file not found for inspection.")
            return

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            print("\n📊 DATABASE SCHEMA ANALYSIS:")
            for table in tables:
                table_name = table[0]
                print(f"\nTable: {table_name}")
                
                # Get columns info
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                for col in columns:
                    # cid, name, type, notnull, dflt_value, pk
                    pk_str = "PK" if col[5] else ""
                    nn_str = "NOT NULL" if col[3] else ""
                    print(f"  - {col[1]} ({col[2]}) {nn_str} {pk_str}")
                
                # Get indices
                cursor.execute(f"PRAGMA index_list({table_name})")
                indices = cursor.fetchall()
                if indices:
                    print("  Indices:")
                    for idx in indices:
                        print(f"  - {idx[1]} (Unique: {idx[2]})")
                else:
                    print("  ⚠️ No indices found (besides PK)")

                # Get FKs
                cursor.execute(f"PRAGMA foreign_key_list({table_name})")
                fks = cursor.fetchall()
                if fks:
                    print("  Foreign Keys:")
                    for fk in fks:
                        # id, seq, table, from, to, on_update, on_delete, match
                        print(f"  - {fk[3]} -> {fk[2]}.{fk[4]} (OnDelete: {fk[6]})")
                else:
                    pass

            conn.close()
        except Exception as e:
            self.add_issue('DATABASE', 'HIGH', 'DB', f"Error inspecting DB: {str(e)}")

    def run(self):
        print("="*60)
        print("RUNNING DEEP AUDIT...")
        print("="*60)

        # 1. Code Analysis
        for file_path in self.project_root.rglob('*.py'):
            if 'venv' in str(file_path) or '__pycache__' in str(file_path):
                continue
                
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                self.check_secrets(file_path, content)
                self.check_error_handling(file_path, content)

        # 2. Database Analysis
        self.check_database_schema()

        # 3. Report
        print("\n" + "="*60)
        print("AUDIT FINDINGS")
        print("="*60)
        
        for issue in self.issues:
            print(f"[{issue['severity']}] {issue['category']} - {issue['file']}")
            print(f"  {issue['message']}")

if __name__ == '__main__':
    auditor = DeepAuditor('c:/asesoriaimss.io')
    auditor.run()
