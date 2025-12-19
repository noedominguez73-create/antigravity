"""
Test script to verify Gemini API is working
"""
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Test 1: Check API key
api_key = os.getenv('GEMINI_API_KEY')
print("="*60)
print("TEST 1: API KEY")
print("="*60)
if api_key:
    print(f"✅ API Key found: {api_key[:20]}...")
else:
    print("❌ API Key NOT found")
    exit(1)

# Test 2: Try to initialize Gemini
print("\n" + "="*60)
print("TEST 2: GEMINI INITIALIZATION")
print("="*60)
try:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    print("✅ Gemini initialized successfully")
    
    # Test 3: Send a test message
    print("\n" + "="*60)
    print("TEST 3: SEND TEST MESSAGE")
    print("="*60)
    response = model.generate_content("Di 'Hola' en una palabra")
    print(f"✅ Response received: {response.text}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

print("\n" + "="*60)
print("✅ ALL TESTS PASSED - Gemini is working!")
print("="*60)
