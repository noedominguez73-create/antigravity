
import google.generativeai as genai
import os
import sys
from dotenv import load_dotenv

load_dotenv()

print(f"Python executing: {sys.executable}")
try:
    import google.generativeai as genai
    print(f"google-generativeai version: {genai.__version__}")
except AttributeError:
    print("Could not determine version (older lib?)")
except ImportError:
    print("Library not installed")
    exit(1)

api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    print("NO API KEY FOUND")
    exit(1)

genai.configure(api_key=api_key)

print("\n--- AVAILABLE MODELS ---")
valid_models = []
try:
    for m in genai.list_models():
        print(f"Name: {m.name}")
        print(f"Methods: {m.supported_generation_methods}")
        if 'generateContent' in m.supported_generation_methods:
            valid_models.append(m.name)
        print("-" * 20)
except Exception as e:
    print(f"Listing failed: {e}")

print(f"\nFound {len(valid_models)} candidate models for generateContent.")

print("\n--- TESTING GENERATION ---")
for model_name in valid_models:
    print(f"Testing {model_name}...")
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Hello, suggest a hair color.")
        print(f"SUCCESS with {model_name}!")
        print(f"Response snippet: {response.text[:50]}...")
        print(f"*** RECOMMENDED MODEL: {model_name} ***")
        with open('working_model.txt', 'w') as f:
            f.write(model_name)
        break
    except Exception as e:
        print(f"FAILED with {model_name}: {e}")

print("\nDone.")
