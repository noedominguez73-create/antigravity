import requests
from app import create_app, db
from app.models import User, Professional, ProfessionalURL

BASE_URL = 'http://localhost:5000'
USER_EMAIL = 'user4@test.com'
PASSWORD = 'password123'

def reproduce_url_bug():
    print("--- Reproducing URL Saving Bug ---")
    
    # 1. Login
    print(f"Logging in as {USER_EMAIL}...")
    response = requests.post(f'{BASE_URL}/api/auth/login', json={
        'email': USER_EMAIL,
        'password': PASSWORD
    })
    
    if response.status_code != 200:
        print(f"Login failed: {response.text}")
        return

    token = response.json()['data']['token']
    user_id = response.json()['data']['user']['id']
    print(f"Login successful. User ID: {user_id}")

    # 2. Get Professional ID (Simulating loadProfessionalData)
    print(f"Getting professional profile for user {user_id}...")
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.get(f'{BASE_URL}/api/profesionales?user_id={user_id}', headers=headers)
    
    data = response.json()
    if not data['success'] or not data['data']['professionals']:
        print("No professional profile found!")
        return
        
    prof_id = data['data']['professionals'][0]['id']
    print(f"Professional ID: {prof_id}")

    # 3. Add URL
    print(f"Adding URL to professional {prof_id}...")
    url_payload = {
        'url': 'https://www.imss.gob.mx/patrones',
        'specialty': 'Patrones',
        'description': 'Test URL for bug reproduction'
    }
    
    response = requests.post(f'{BASE_URL}/api/profesionales/{prof_id}/urls', json=url_payload, headers=headers)
    print(f"Add URL Status: {response.status_code}")
    print(f"Add URL Response: {response.text}")
    
    if response.status_code == 201:
        print("URL added successfully via API.")
        
        # 4. Verify in DB
        app = create_app()
        with app.app_context():
            url_obj = ProfessionalURL.query.filter_by(url='https://www.imss.gob.mx/patrones').order_by(ProfessionalURL.id.desc()).first()
            if url_obj:
                print(f"DB Verification: Found URL ID {url_obj.id}")
                print(f"  - Professional ID: {url_obj.professional_id}")
                print(f"  - URL: {url_obj.url}")
                
                if url_obj.professional_id == prof_id:
                    print("SUCCESS: URL saved with correct professional_id.")
                else:
                    print(f"FAILURE: URL saved with WRONG professional_id! Expected {prof_id}, got {url_obj.professional_id}")
            else:
                print("FAILURE: URL not found in DB!")
    else:
        print("FAILURE: API returned error.")

if __name__ == '__main__':
    reproduce_url_bug()
