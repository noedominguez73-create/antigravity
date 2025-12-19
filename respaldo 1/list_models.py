import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    print("Error: GEMINI_API_KEY not found.")
    exit(1)

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"

try:
    response = requests.get(url)
    with open('models_list.txt', 'w') as f:
        if response.status_code == 200:
            models = response.json().get('models', [])
            f.write(f"Found {len(models)} models:\n")
            for m in models:
                if 'generateContent' in m.get('supportedGenerationMethods', []):
                    f.write(f"- {m['name']}\n")
        else:
            f.write(f"Error: {response.status_code} - {response.text}\n")
    print("Done writing to models_list.txt")
except Exception as e:
    print(f"Exception: {e}")
