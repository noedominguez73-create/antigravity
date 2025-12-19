import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
genai.configure(api_key=api_key)

print("Listing Models...")
try:
    for m in genai.list_models():
        print(f"Name: {m.name}")
        print(f"Supported Generative Methods: {m.supported_generation_methods}")
        print("---")
except Exception as e:
    print(f"Error: {e}")
