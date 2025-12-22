import requests
import json

url = 'http://127.0.0.1:5000/api/auth/login'
headers = {'Content-Type': 'application/json'}
payload = {
    'email': 'user_e446e3ee@completmirror.io',
    'password': '123456'
}

try:
    response = requests.post(url, headers=headers, json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
