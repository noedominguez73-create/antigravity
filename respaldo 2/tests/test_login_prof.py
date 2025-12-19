import requests
import json

url = 'http://127.0.0.1:5000/api/auth/login'
data = {
    'email': 'prof@test.com',
    'password': 'password123'
}

if __name__ == "__main__":
    try:
        response = requests.post(url, json=data)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("Login Successful!")
            print(json.dumps(response.json(), indent=2))
        else:
            print("Login Failed!")
            print(response.text)
            
    except Exception as e:
        print(f"Error: {str(e)}")
