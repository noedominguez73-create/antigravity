import requests
import json

# 1. Login to get token
login_url = 'http://127.0.0.1:5000/api/auth/login'
login_data = {
    'email': 'prof@test.com',
    'password': 'password123'
}

try:
    session = requests.Session()
    resp = session.post(login_url, json=login_data)
    token = resp.json()['data']['token']
    user_id = resp.json()['data']['user']['id']
    print(f"Logged in. User ID: {user_id}")

    # 2. Get Professional ID (assuming 1 for now, or fetch from profile)
    # In a real scenario we'd fetch the profile, but let's assume ID 1 based on creation order
    prof_id = 1 
    
    # 3. Test POST URL
    url_endpoint = f'http://127.0.0.1:5000/api/chatbot/{prof_id}/urls'
    headers = {'Authorization': f'Bearer {token}'}
    payload = {
        "url": "https://www.imss.gob.mx/personas-trabajadoras-independientes",
        "specialty": "Calculadora Modalidad 10",
        "description": "Test URL from script"
    }
    
    print(f"Testing POST to {url_endpoint}")
    resp = session.post(url_endpoint, json=payload, headers=headers)
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.text}")
    
    # 4. Verify it's in the list
    print(f"Testing GET from {url_endpoint}")
    resp = session.get(url_endpoint, headers=headers)
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.text}")

except Exception as e:
    print(f"Error: {e}")
