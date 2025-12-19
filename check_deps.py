try:
    import openpyxl
    print("✅ openpyxl installed")
except ImportError as e:
    print(f"❌ openpyxl MISSING: {e}")

try:
    import PyPDF2
    print("✅ PyPDF2 installed")
except ImportError as e:
    print(f"❌ PyPDF2 MISSING: {e}")

try:
    import docx
    print("✅ python-docx installed")
except ImportError as e:
    print(f"❌ python-docx MISSING: {e}")
