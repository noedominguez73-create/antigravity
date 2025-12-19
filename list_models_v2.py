import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
genai.configure(api_key=api_key)

print("Available Models:")
for m in genai.list_models():
    print(f"- {m.name} ({m.supported_generation_methods})")
