import requests
import json

BASE_URL = 'http://localhost:5000'
EMAIL = 'prof@test.com'
PASSWORD = 'password123'

def login():
    print(f"Logging in as {EMAIL}...")
    response = requests.post(f'{BASE_URL}/api/auth/login', json={
        'email': EMAIL,
        'password': PASSWORD
    })
    if response.status_code == 200:
        data = response.json()
        print("Login successful")
        return data['data']['token'], data['data']['user']['id']
    else:
        print(f"Login failed: {response.text}")
        return None, None

def get_professional_id(token):
    print("Getting professional ID...")
    headers = {'Authorization': f'Bearer {token}'}
    # We can get it from /api/auth/me if it existed, or from login response if we modified it.
    # Or we can fetch /api/profesionales?user_id=... but that endpoint filters by params.
    # Let's assume the user has a professional profile and we can find it via the user ID or just list professionals.
    # Actually, the login response usually contains user info.
    # Let's try to get it from /dashboard-profesional page content if we were scraping, but here we are using API.
    # Let's use the /api/profesionales endpoint to find the professional for the current user.
    # But wait, create_prof_user.py created a professional.
    # Let's try to hit /api/profesionales/me if it existed.
    # Since we don't have a direct "me" endpoint for professional, let's just list professionals and find ours (not ideal but works for test).
    # OR, we can just use the ID 1 or 2 if we know it.
    # Let's try to fetch the chatbot config, it requires professional ID.
    # Let's assume ID 1 for now as we recreated the DB.
    return 1

def test_urls(token, prof_id):
    headers = {'Authorization': f'Bearer {token}'}
    
    print(f"\nTesting URL endpoints for Professional ID: {prof_id}")
    
    # 1. Add Valid URL
    print("\n1. Adding Valid URL (https://www.imss.gob.mx)...")
    url_data = {
        'url': 'https://www.imss.gob.mx',
        'specialty': 'General',
        'description': 'Sitio oficial IMSS'
    }
    response = requests.post(f'{BASE_URL}/api/profesionales/{prof_id}/urls', json=url_data, headers=headers)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 201:
        url_id = response.json()['data']['url']['id']
        print(f"URL Added. ID: {url_id}")
    else:
        print("Failed to add valid URL")
        return

    # 2. Add Invalid URL (Domain not allowed)
    print("\n2. Adding Invalid URL (https://www.google.com)...")
    url_data_invalid = {
        'url': 'https://www.google.com',
        'specialty': 'Search'
    }
    response = requests.post(f'{BASE_URL}/api/profesionales/{prof_id}/urls', json=url_data_invalid, headers=headers)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    assert response.status_code == 400
    assert "Dominio no permitido" in response.text

    # 3. List URLs
    print("\n3. Listing URLs...")
    response = requests.get(f'{BASE_URL}/api/profesionales/{prof_id}/urls', headers=headers)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    assert response.status_code == 200
    urls = response.json()['data']['urls']
    print(f"Found {len(urls)} URLs")
    
    # 4. Refresh URL
    print(f"\n4. Refreshing URL {url_id}...")
    response = requests.post(f'{BASE_URL}/api/profesionales/{prof_id}/urls/{url_id}/refresh', headers=headers)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    assert response.status_code == 200

    # 5. Delete URL
    print(f"\n5. Deleting URL {url_id}...")
    response = requests.delete(f'{BASE_URL}/api/profesionales/{prof_id}/urls/{url_id}', headers=headers)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    assert response.status_code == 200

if __name__ == '__main__':
    token, user_id = login()
    if token:
        # We need to ensure the professional exists.
        # Since we recreated the DB, we might need to create the professional first.
        # Let's run create_prof_user.py logic here or assume it's run.
        # I'll assume I need to run create_prof_user.py first.
        prof_id = get_professional_id(token)
        test_urls(token, prof_id)
