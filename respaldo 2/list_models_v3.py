import google.generativeai as genai
import os
from dotenv import load_dotenv
import sys

# Force UTF-8 for stdout
sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
genai.configure(api_key=api_key)

print("--- START MODEL LIST ---")
try:
    for m in genai.list_models():
        print(f"{m.name}")
except Exception as e:
    print(f"Error listing models: {e}")
print("--- END MODEL LIST ---")
