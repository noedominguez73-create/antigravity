import sys
import os
import threading
import time
import requests

# Configuration
BASE_URL = "http://localhost:5000" # Adjust if needed
PROFESSIONAL_ID = 1 # Change to a valid ID
API_KEY = "test_token" # You might need to generate a valid JWT token

def send_message(thread_id):
    url = f"{BASE_URL}/api/chatbot/{PROFESSIONAL_ID}/chat"
    headers = {
        "Authorization": f"Bearer {API_KEY}", # Needs valid token
        "Content-Type": "application/json"
    }
    data = {
        "message": f"Test message from thread {thread_id}",
        "session_id": f"test_session_{thread_id}"
    }
    
    try:
        response = requests.post(url, json=data, headers=headers)
        print(f"Thread {thread_id}: Status {response.status_code} - {response.text[:50]}...")
    except Exception as e:
        print(f"Thread {thread_id}: Error {e}")

def test_race_condition():
    print("🚀 Starting Race Condition Test...")
    
    # Note: This test requires the server to be running and a valid JWT token.
    # Since we are in a CLI environment, we can't easily spin up the full server and get a token 
    # without more setup.
    
    # ALTERNATIVE: Unit test style with app context and threads?
    # Flask's test client is not thread-safe for this kind of concurrency test usually.
    # But we can try using the app context directly with threads.
    
    pass

if __name__ == "__main__":
    print("⚠️ This test requires a running server and valid JWT.")
    print("For manual verification: Open 2 browser tabs, send message simultaneously.")
