import requests
from app import create_app, db
from app.models import User, Professional

BASE_URL = 'http://localhost:5000'
EMAIL = 'prof@test.com'
PASSWORD = 'password123'

def reproduce():
    # 1. Login
    print(f"Logging in as {EMAIL}...")
    response = requests.post(f'{BASE_URL}/api/auth/login', json={
        'email': EMAIL,
        'password': PASSWORD
    })
    
    if response.status_code != 200:
        print(f"Login failed: {response.text}")
        return

    data = response.json()
    token = data['data']['token']
    user_id = data['data']['user']['id']
    print(f"Login successful. User ID: {user_id}")

    # 2. Get Professional ID
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.get(f'{BASE_URL}/api/profesionales?user_id={user_id}', headers=headers)
    prof_data = response.json()['data']['professionals'][0]
    prof_id = prof_data['id']
    prof_user_id = prof_data['user_id']
    print(f"Professional ID: {prof_id}, Owner User ID: {prof_user_id}")

    # 3. Verify IDs match
    print(f"User ID type: {type(user_id)}")
    print(f"Prof Owner ID type: {type(prof_user_id)}")
    
    if user_id != prof_user_id:
        print(f"MISMATCH: Logged in User ID ({user_id}) != Professional Owner ID ({prof_user_id})")
    else:
        print("IDs match.")

    # 4. Attempt Update
    print(f"Attempting to update profile {prof_id}...")
    update_data = {
        'bio': 'Updated bio for testing permissions'
    }
    response = requests.put(f'{BASE_URL}/api/profesionales/{prof_id}', json=update_data, headers=headers)
    print(f"Update Status: {response.status_code}")
    print(f"Update Response: {response.text}")

if __name__ == '__main__':
    reproduce()
