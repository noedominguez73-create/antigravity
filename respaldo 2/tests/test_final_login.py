import requests
import json

url = 'http://127.0.0.1:5000/api/auth/login'
data = {
    'email': 'juan.perez@example.com',
    'password': 'password123'
}

print("=" * 60)
print("Testing Login Endpoint")
print("=" * 60)
print(f"URL: {url}")
print(f"Data: {json.dumps(data, indent=2)}")
print()

try:
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Content-Type: {response.headers.get('Content-Type')}")
    print()
    
    if response.status_code == 200:
        print("[SUCCESS] Login successful!")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"[ERROR] Login failed!")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"[ERROR] Request failed: {str(e)}")

print("=" * 60)
