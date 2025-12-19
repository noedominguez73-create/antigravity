
import requests

url = "https://www.imss.gob.mx/personas-trabajadoras-independientes/calculadora"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

try:
    response = requests.get(url, headers=headers, timeout=10)
    print(f"Status Code: {response.status_code}")
    print("--- Raw HTML Start ---")
    print(response.text[:2000])
    print("--- Raw HTML End ---")
    
except Exception as e:
    print(f"Error: {e}")
