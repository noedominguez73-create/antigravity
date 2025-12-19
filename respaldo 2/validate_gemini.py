import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for API key
api_key = os.getenv('GEMINI_API_KEY')

print("="*60)
print("VALIDACIÓN DE GEMINI API KEY")
print("="*60)

if api_key:
    print(f"✅ API Key encontrada")
    print(f"   Primeros 20 caracteres: {api_key[:20]}...")
    print(f"   Longitud total: {len(api_key)} caracteres")
    
    # Validate format
    if api_key.startswith('AIza'):
        print("✅ Formato correcto (comienza con 'AIza')")
    else:
        print("⚠️  Formato inusual (no comienza con 'AIza')")
        
    # Check for spaces
    if ' ' in api_key:
        print("❌ ADVERTENCIA: La API key contiene espacios")
    else:
        print("✅ Sin espacios")
        
else:
    print("❌ API Key NO encontrada en .env")
    print("   Variable esperada: GEMINI_API_KEY")
    print("   Verifica que el archivo .env existe y contiene:")
    print("   GEMINI_API_KEY=tu_api_key_aqui")

print("="*60)
