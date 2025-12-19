import requests
from app import create_app, db
from app.models import User
from werkzeug.security import generate_password_hash

BASE_URL = 'http://localhost:5000'
ATTACKER_EMAIL = 'attacker@test.com'
PASSWORD = 'password123'

def create_attacker_user():
    app = create_app()
    with app.app_context():
        user = User.query.filter_by(email=ATTACKER_EMAIL).first()
        if not user:
            user = User(
                email=ATTACKER_EMAIL,
                password_hash=generate_password_hash(PASSWORD),
                role='professional',
                full_name='Attacker User'
            )
            db.session.add(user)
            db.session.commit()
            print("Attacker user created.")
        else:
            print("Attacker user already exists.")

def reproduce_mismatch():
    # 1. Create attacker user
    create_attacker_user()

    # 2. Login as attacker
    print(f"Logging in as {ATTACKER_EMAIL}...")
    response = requests.post(f'{BASE_URL}/api/auth/login', json={
        'email': ATTACKER_EMAIL,
        'password': PASSWORD
    })
    
    if response.status_code != 200:
        print(f"Login failed: {response.text}")
        return

    data = response.json()
    token = data['data']['token']
    attacker_id = data['data']['user']['id']
    print(f"Login successful. Attacker User ID: {attacker_id}")

    # 3. Try to update Professional ID 1 (owned by User ID 1)
    target_prof_id = 1
    print(f"Attempting to update TARGET profile {target_prof_id} as ATTACKER...")
    
    headers = {'Authorization': f'Bearer {token}'}
    update_data = {
        'bio': 'HACKED BIO'
    }
    response = requests.put(f'{BASE_URL}/api/profesionales/{target_prof_id}', json=update_data, headers=headers)
    print(f"Update Status: {response.status_code}")
    print(f"Update Response: {response.text}")
    
    if response.status_code == 403:
        print("SUCCESS: Permission denied as expected.")
    else:
        print("FAILURE: Operation was not denied!")

if __name__ == '__main__':
    reproduce_mismatch()
