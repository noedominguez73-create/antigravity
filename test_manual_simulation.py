import requests
import os
import json

BASE_URL = 'http://localhost:5000'
EMAIL = 'prof@test.com'
PASSWORD = 'password123'

# Create dummy files
with open('test_doc.txt', 'w') as f:
    f.write('This is a test document content for knowledge base.')

with open('test_doc.pdf', 'wb') as f:
    f.write(b'%PDF-1.4\n%...') # Minimal dummy PDF header

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

def get_professional_id(token, user_id):
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.get(f'{BASE_URL}/api/profesionales?user_id={user_id}', headers=headers)
    if response.status_code == 200:
        profs = response.json()['data']['professionals']
        if profs:
            return profs[0]['id']
    return None

def test_documents(token, prof_id):
    headers = {'Authorization': f'Bearer {token}'}
    print("\n=== Testing Documents ===")
    
    # 1. Upload TXT
    print("Uploading TXT...")
    files = {'file': ('test_doc.txt', open('test_doc.txt', 'rb'), 'text/plain')}
    response = requests.post(f'{BASE_URL}/api/chatbot/{prof_id}/knowledge-base/upload', files=files, headers=headers)
    print(f"Upload TXT Status: {response.status_code}")
    if response.status_code == 201:
        txt_id = response.json()['data']['document']['id']
        print(f"TXT Uploaded. ID: {txt_id}")
    else:
        print(f"Upload failed: {response.text}")
        return

    # 2. List Documents
    print("Listing Documents...")
    response = requests.get(f'{BASE_URL}/api/chatbot/{prof_id}/knowledge-base/documents', headers=headers)
    print(f"List Status: {response.status_code}")
    docs = response.json()['data']['documents']
    print(f"Found {len(docs)} documents")
    
    # 3. Delete Document
    print(f"Deleting Document {txt_id}...")
    response = requests.delete(f'{BASE_URL}/api/chatbot/{prof_id}/knowledge-base/documents/{txt_id}', headers=headers)
    print(f"Delete Status: {response.status_code}")

def test_urls(token, prof_id):
    headers = {'Authorization': f'Bearer {token}'}
    print("\n=== Testing URLs ===")
    
    # 1. Add Valid URL
    print("Adding Valid URL...")
    url_data = {
        'url': 'https://www.imss.gob.mx',
        'specialty': 'General',
        'description': 'Sitio oficial IMSS'
    }
    response = requests.post(f'{BASE_URL}/api/profesionales/{prof_id}/urls', json=url_data, headers=headers)
    print(f"Add URL Status: {response.status_code}")
    if response.status_code == 201:
        url_id = response.json()['data']['url']['id']
        print(f"URL Added. ID: {url_id}")
    else:
        print(f"Add URL failed: {response.text}")
        return

    # 2. Refresh URL
    print(f"Refreshing URL {url_id}...")
    response = requests.post(f'{BASE_URL}/api/profesionales/{prof_id}/urls/{url_id}/refresh', headers=headers)
    print(f"Refresh Status: {response.status_code}")

    # 3. Delete URL
    print(f"Deleting URL {url_id}...")
    response = requests.delete(f'{BASE_URL}/api/profesionales/{prof_id}/urls/{url_id}', headers=headers)
    print(f"Delete Status: {response.status_code}")

def test_chatbot(token, prof_id):
    headers = {'Authorization': f'Bearer {token}'}
    print("\n=== Testing Chatbot (Preview) ===")
    
    msg_data = {'message': 'Hola, prueba de sistema'}
    response = requests.post(f'{BASE_URL}/api/chatbot/{prof_id}/mensaje', json=msg_data, headers=headers)
    print(f"Chat Status: {response.status_code}")
    print(f"Response: {response.text}")
    if response.status_code == 200:
        print("Chatbot test successful!")
    else:
        print("Chatbot test failed.")

if __name__ == '__main__':
    token, user_id = login()
    if token:
        prof_id = get_professional_id(token, user_id)
        if prof_id:
            print(f"Professional ID: {prof_id}")
            test_documents(token, prof_id)
            test_urls(token, prof_id)
            test_chatbot(token, prof_id)
        else:
            print("Professional not found")
