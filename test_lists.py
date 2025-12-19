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

    # Assume prof_id = 1
    prof_id = 1 
    headers = {'Authorization': f'Bearer {token}'}
    
    # 2. Test GET URLs
    url_endpoint = f'http://127.0.0.1:5000/api/chatbot/{prof_id}/urls'
    print(f"Testing GET URLs from {url_endpoint}")
    resp = session.get(url_endpoint, headers=headers)
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.text}")

    # 3. Test GET Documents
    doc_endpoint = f'http://127.0.0.1:5000/api/chatbot/{prof_id}/knowledge-base/documents'
    print(f"Testing GET Documents from {doc_endpoint}")
    resp = session.get(doc_endpoint, headers=headers)
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.text}")

except Exception as e:
    print(f"Error: {e}")
