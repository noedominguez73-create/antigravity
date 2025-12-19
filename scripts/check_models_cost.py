import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
genai.configure(api_key=api_key)

print("Listing models...")
for m in genai.list_models():
    print(f"Name: {m.name}")
    print(f"Methods: {m.supported_generation_methods}")
    print("-" * 20)
