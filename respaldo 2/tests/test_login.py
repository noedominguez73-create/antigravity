"""
Test script to verify login endpoint
"""
import requests
import json

url = 'http://127.0.0.1:5000/api/auth/login'
data = {
    'email': 'juan.perez@example.com',
    'password': 'password123'
}

if __name__ == "__main__":
    try:
        response = requests.post(url, json=data)
        print(f"Status Code: {response.status_code}")
        print(f"Headers: {response.headers.get('Content-Type')}")
        print(f"Response Text: {response.text[:500]}")
        
        if response.headers.get('Content-Type', '').startswith('application/json'):
            print(f"JSON Response: {json.dumps(response.json(), indent=2)}")
        else:
            print("ERROR: Response is not JSON!")
            print(f"Full Response:\n{response.text}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
