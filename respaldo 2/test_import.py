
print("Starting import...")
try:
    from app.services.rag_service import rag_service
    print("Import successful!")
    print(f"Service instance: {rag_service}")
    print(f"Method: {rag_service.retrieve_context}")
except Exception as e:
    print(f"Import failed: {e}")
