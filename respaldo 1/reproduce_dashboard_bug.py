import requests
from app import create_app, db
from app.models import User, Professional
from werkzeug.security import generate_password_hash

BASE_URL = 'http://localhost:5000'
USER_EMAIL = 'user4@test.com'
PASSWORD = 'password123'

def setup_test_data():
    app = create_app()
    with app.app_context():
        # Ensure User 1 exists and has a profile
        user1 = User.query.get(1)
        if not user1:
            print("User 1 missing, creating...")
            user1 = User(id=1, email='user1@test.com', password_hash=generate_password_hash(PASSWORD), role='professional', full_name='User One')
            db.session.add(user1)
            db.session.commit()
            
        prof1 = Professional.query.filter_by(user_id=1).first()
        if not prof1:
            print("Profile 1 missing, creating...")
            prof1 = Professional(user_id=1, specialty='Spec 1', city='City 1', bio='Bio 1')
            db.session.add(prof1)
            db.session.commit()

        # Create User 4 and their profile
        user4 = User.query.filter_by(email=USER_EMAIL).first()
        if not user4:
            print("User 4 missing, creating...")
            user4 = User(email=USER_EMAIL, password_hash=generate_password_hash(PASSWORD), role='professional', full_name='User Four')
            db.session.add(user4)
            db.session.commit()
            
        prof4 = Professional.query.filter_by(user_id=user4.id).first()
        if not prof4:
            print(f"Profile for User {user4.id} missing, creating...")
            prof4 = Professional(user_id=user4.id, specialty='Spec 4', city='City 4', bio='Bio 4')
            db.session.add(prof4)
            db.session.commit()
            
        return user4.id

def reproduce_bug():
    user4_id = setup_test_data()
    print(f"Testing with User ID: {user4_id}")

    # 1. Login as User 4
    print(f"Logging in as {USER_EMAIL}...")
    response = requests.post(f'{BASE_URL}/api/auth/login', json={
        'email': USER_EMAIL,
        'password': PASSWORD
    })
    
    if response.status_code != 200:
        print(f"Login failed: {response.text}")
        return

    token = response.json()['data']['token']

    # 2. Call GET /profesionales?user_id=4
    print(f"Calling GET /profesionales?user_id={user4_id}...")
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.get(f'{BASE_URL}/api/profesionales?user_id={user4_id}', headers=headers)
    
    data = response.json()
    profs = data['data']['professionals']
    
    print(f"Found {len(profs)} professionals.")
    
    if len(profs) > 0:
        first_prof = profs[0]
        print(f"First Professional ID: {first_prof['id']}, Owner User ID: {first_prof['user_id']}")
        
        if first_prof['user_id'] != user4_id:
            print("BUG REPRODUCED: Returned profile does NOT belong to the requested user_id!")
            print(f"Expected User ID: {user4_id}, Got: {first_prof['user_id']}")
        else:
            print("Test Passed: Returned profile belongs to the requested user_id.")
    else:
        print("No professionals found.")

if __name__ == '__main__':
    reproduce_bug()
