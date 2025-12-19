
import requests
import json

# Configuration
BASE_URL = 'http://localhost:5000'
PROFESSIONAL_ID = 3 # From screenshot
TOKEN = '' # Will need to login first or use a hardcoded token if available

def login():
    print("Logging in...")
    try:
        # Try to login as admin or user to get token
        # Using a known test user if possible, or just print instructions
        # For this script, I'll assume I can use the admin login I saw earlier or just fail if no creds
        # Let's try to find a user in the DB or just use a placeholder and ask user to fill it
        # Actually, I can use the 'login' endpoint if I knew a user.
        # Let's try 'dominguez73@gmail.com' from previous screenshots?
        # Password? I don't know it.
        # I will try to use a hardcoded token if I can find one in the logs or just skip auth if possible (but endpoints are protected)
        
        # Alternative: Use the 'debug' blueprint if available to get a token? No.
        # I will try to use the 'register' endpoint to create a temp user?
        
        # Let's try to register a temp user
        email = f"test_debug_{import_uuid().hex[:8]}@example.com"
        password = "password123"
        
        resp = requests.post(f"{BASE_URL}/api/auth/register", json={
            "email": email,
            "password": password,
            "full_name": "Debug User",
            "role": "user"
        })
        
        if resp.status_code == 201:
            return resp.json()['data']['token']
        
        # If fail, try login with the email (maybe already exists)
        resp = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": email,
            "password": password
        })
        
        if resp.status_code == 200:
            return resp.json()['data']['token']
            
        print("Failed to get token.")
        return None
        
    except Exception as e:
        print(f"Login error: {e}")
        return None

def import_uuid():
    import uuid
    return uuid.uuid4()

def reproduce():
    token = login()
    if not token:
        print("Skipping reproduction due to no token.")
        return

    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }

    print(f"\nTesting Chatbot for Professional {PROFESSIONAL_ID}...")

    # 1. First Message
    print("\n1. Sending First Message...")
    payload1 = {
        "message": "Hola, configura tu chatbot",
        "professional_id": PROFESSIONAL_ID
    }
    
    try:
        resp1 = requests.post(f"{BASE_URL}/api/chatbot/{PROFESSIONAL_ID}/chat", json=payload1, headers=headers)
        print(f"Status: {resp1.status_code}")
        print(f"Response: {resp1.text[:200]}...")
        
        if resp1.status_code != 200:
            print("Failed step 1.")
            return

        data1 = resp1.json()
        session_id = data1['data']['session_id']
        print(f"Session ID: {session_id}")

        # 2. Second Message (With Session ID)
        print("\n2. Sending Second Message (Triggering RAG)...")
        payload2 = {
            "message": "que dice el articulo 10 de la ley del isr",
            "professional_id": PROFESSIONAL_ID,
            "session_id": session_id
        }
        
        resp2 = requests.post(f"{BASE_URL}/api/chatbot/{PROFESSIONAL_ID}/chat", json=payload2, headers=headers)
        print(f"Status: {resp2.status_code}")
        print(f"Response: {resp2.text}") # Print full response to see error
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    reproduce()
