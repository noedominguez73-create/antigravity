import logging
import json
from app import create_app, db
from app.models import User

# Setup app
app = create_app()
app.config['TESTING'] = True
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'

def test_auth_me():
    with app.app_context():
        db.create_all()
        
        # Create test client
        client = app.test_client()
        
        # 1. Register user
        reg_data = {
            'email': 'me_test@example.com',
            'password': 'Password123!',
            'full_name': 'Me Tester'
        }
        client.post('/api/auth/registro', json=reg_data)
        
        # 2. Login to get token
        login_data = {
            'email': 'me_test@example.com',
            'password': 'Password123!'
        }
        login_res = client.post('/api/auth/login', json=login_data)
        token = login_res.get_json()['data']['token']
        
        print(f"Token obtained: {token[:10]}...")
        
        # 3. Request /me
        headers = {
            'Authorization': f'Bearer {token}'
        }
        me_res = client.get('/api/auth/me', headers=headers)
        me_data = me_res.get_json()
        
        print(f"Status Code: {me_res.status_code}")
        print(f"Response: {me_data}")
        
        if me_res.status_code == 200 and me_data['data']['user']['email'] == 'me_test@example.com':
            print("✅ Endpoint /me SUCCESS")
        else:
            print("❌ Endpoint /me FAILED")

if __name__ == "__main__":
    test_auth_me()
