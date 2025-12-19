# Verificar que Gemini funciona
from dotenv import load_dotenv
import google.generativeai as genai
import os

load_dotenv()

api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_GEMINI_API_KEY')
print(f"API Key found: {'Yes' if api_key else 'No'}")

if not api_key:
    print("Error: No API key found in environment variables.")
    exit(1)

try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content('Hola, responde solo con la palabra "Funcionando"')
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error testing Gemini: {e}")
