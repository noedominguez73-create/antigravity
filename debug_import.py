import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

try:
    import app.models
    print("Successfully imported app.models")
    print("Attributes in app.models:")
    for attr in dir(app.models):
        if not attr.startswith('__'):
            print(f"- {attr}")
            
    from app.models import KnowledgeBaseDocument
    print("Successfully imported KnowledgeBaseDocument")
except Exception as e:
    print(f"Error: {e}")
