"""
Comprehensive Code Audit Script for AsesoriaIMSS.io
Validates imports, structure, and identifies issues
"""

import os
import sys
import ast
import importlib.util
from pathlib import Path

class CodeAuditor:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.issues = {
            'critical': [],
            'warning': [],
            'info': []
        }
        self.stats = {
            'total_files': 0,
            'total_lines': 0,
            'import_errors': 0,
            'syntax_errors': 0
        }
    
    def add_issue(self, level, category, file, message):
        """Add an issue to the report"""
        self.issues[level].append({
            'category': category,
            'file': str(file),
            'message': message
        })
    
    def check_imports(self, file_path):
        """Check if all imports in a file are valid"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content, filename=str(file_path))
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        try:
                            importlib.import_module(alias.name)
                        except ImportError as e:
                            self.add_issue('warning', 'IMPORT', file_path, 
                                         f"Cannot import '{alias.name}': {e}")
                            self.stats['import_errors'] += 1
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        try:
                            importlib.import_module(node.module)
                        except ImportError as e:
                            self.add_issue('warning', 'IMPORT', file_path,
                                         f"Cannot import from '{node.module}': {e}")
                            self.stats['import_errors'] += 1
            
            return True
        
        except SyntaxError as e:
            self.add_issue('critical', 'SYNTAX', file_path, f"Syntax error: {e}")
            self.stats['syntax_errors'] += 1
            return False
        
        except Exception as e:
            self.add_issue('warning', 'PARSE', file_path, f"Parse error: {e}")
            return False
    
    def check_file_structure(self):
        """Check project structure"""
        required_files = [
            'run.py',
            'requirements.txt',
            '.env.example',
            'app/__init__.py',
            'app/models.py'
        ]
        
        for file in required_files:
            file_path = self.project_root / file
            if not file_path.exists():
                self.add_issue('critical', 'STRUCTURE', file,
                             f"Required file missing: {file}")
    
    def find_orphaned_files(self):
        """Find Python files that might be orphaned"""
        app_dir = self.project_root / 'app'
        
        # Check for __pycache__ in wrong places
        for pycache in self.project_root.rglob('__pycache__'):
            if pycache.parent == self.project_root:
                self.add_issue('info', 'CLEANUP', pycache,
                             "__pycache__ in root directory")
        
        # Check for test files in wrong locations
        test_files = list(self.project_root.glob('test_*.py'))
        if len(test_files) > 5:
            self.add_issue('info', 'STRUCTURE', 'root',
                         f"Too many test files in root ({len(test_files)}). Consider tests/ folder")
    
    def count_lines(self, file_path):
        """Count lines in a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return len(f.readlines())
        except:
            return 0
    
    def audit_all_python_files(self):
        """Audit all Python files in the project"""
        print("="*60)
        print("PASO 1: ANÁLISIS DE ESTRUCTURA Y ARCHIVOS")
        print("="*60)
        
        # Check structure
        self.check_file_structure()
        
        # Find all Python files
        python_files = list(self.project_root.rglob('*.py'))
        
        # Exclude venv and __pycache__
        python_files = [f for f in python_files 
                       if '.venv' not in str(f) 
                       and 'venv_test' not in str(f)
                       and '__pycache__' not in str(f)]
        
        self.stats['total_files'] = len(python_files)
        
        print(f"\n📁 Archivos Python encontrados: {len(python_files)}")
        print("\nValidando imports y sintaxis...")
        
        for file_path in python_files:
            lines = self.count_lines(file_path)
            self.stats['total_lines'] += lines
            
            # Check imports
            self.check_imports(file_path)
        
        # Check for orphaned files
        self.find_orphaned_files()
        
        print(f"✅ Análisis completado")
        print(f"   Total líneas de código: {self.stats['total_lines']}")
        print(f"   Errores de sintaxis: {self.stats['syntax_errors']}")
        print(f"   Errores de import: {self.stats['import_errors']}")
    
    def generate_report(self):
        """Generate audit report"""
        print("\n" + "="*60)
        print("REPORTE DE AUDITORÍA")
        print("="*60)
        
        # Critical issues
        if self.issues['critical']:
            print(f"\n🔴 CRÍTICO ({len(self.issues['critical'])} issues):")
            for issue in self.issues['critical']:
                print(f"   [{issue['category']}] {issue['file']}")
                print(f"      {issue['message']}")
        
        # Warnings
        if self.issues['warning']:
            print(f"\n⚠️  ADVERTENCIAS ({len(self.issues['warning'])} issues):")
            for issue in self.issues['warning'][:10]:  # Show first 10
                print(f"   [{issue['category']}] {issue['file']}")
                print(f"      {issue['message']}")
            if len(self.issues['warning']) > 10:
                print(f"   ... y {len(self.issues['warning']) - 10} más")
        
        # Info
        if self.issues['info']:
            print(f"\n💡 INFO ({len(self.issues['info'])} items):")
            for issue in self.issues['info']:
                print(f"   [{issue['category']}] {issue['file']}")
                print(f"      {issue['message']}")
        
        # Summary
        print("\n" + "="*60)
        print("RESUMEN")
        print("="*60)
        print(f"Total archivos Python: {self.stats['total_files']}")
        print(f"Total líneas de código: {self.stats['total_lines']}")
        print(f"Issues críticos: {len(self.issues['critical'])}")
        print(f"Advertencias: {len(self.issues['warning'])}")
        print(f"Info: {len(self.issues['info'])}")

if __name__ == '__main__':
    auditor = CodeAuditor('c:/asesoriaimss.io')
    auditor.audit_all_python_files()
    auditor.generate_report()
