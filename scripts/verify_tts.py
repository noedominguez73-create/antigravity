import os
import requests
import json
from dotenv import load_dotenv

# Load env
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')

if not api_key:
    print("❌ ERROR: No API Key found in .env")
    exit(1)

print(f"🔑 Testing with API Key: {api_key[:5]}...{api_key[-5:]}")

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={api_key}"

text = "Hola, esta es una prueba de voz."
voice_style = "Puck"

payload = {
    "contents": [{
        "parts": [{ "text": text }]
    }],
    "generationConfig": {
        "responseModalities": ["AUDIO"],
        "speechConfig": {
            "voiceConfig": {
                "prebuiltVoiceConfig": {
                    "voiceName": voice_style
                }
            }
        }
    }
}

print(f"📡 Sending request to {url}...")
try:
    response = requests.post(url, json=payload, headers={'Content-Type': 'application/json'})
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        # Check structure
        try:
            audio_data = data['candidates'][0]['content']['parts'][0]['inlineData']['data']
            print(f"✅ SUCCESS! Received {len(audio_data)} bytes of audio data.")
        except Exception as e:
            print(f"⚠️ Response 200 but parsing failed: {e}")
            print(json.dumps(data, indent=2))
    else:
        print(f"❌ FAILED: {response.text}")

except Exception as e:
    print(f"❌ EXCEPTION: {e}")
