import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')

if not api_key:
    print("❌ ERROR: No API Key")
    exit(1)

genai.configure(api_key=api_key)

print("Testing SDK google-generativeai...")

try:
    # Try the standard text generation first to confirm key
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content("Hello")
    print(f"✅ Text Gen Works: {response.text}")
    
    # Now try Speech/Audio with Gemini 2.0 Flash Exp
    # Note: As of Dec 2025, pure TTS via Gemini API might be under a different method or client library.
    # However, let's try the generate_content with default settings for the experimental model
    # The 'voice' feature might be specific to the live API or require specific config
    
    # Let's try the specific documented way for Audio Generation if available in 0.8.3
    # If not, we might have to fallback to standard Google Cloud TTS if the user's project supports not just Gemini.
    
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents="Hello, this is a test.",
        config={
            "response_modalities": ["AUDIO"],
            "speech_config": {
                "voice_config": {"prebuilt_voice_config": {"voice_name": "Puck"}}
            }
        }
    )
    
    # Extract audio
    # The SDK usually returns a confirmation object, verify structure
    print("✅ SDK Request sent.")
    
    # Check for binary data
    for part in response.candidates[0].content.parts:
        if part.inline_data:
            print(f"✅ SUCCESS: Received Audio Bytes: {len(part.inline_data.data)}")
            break
            
except Exception as e:
    print(f"❌ SDK ERROR: {e}")
