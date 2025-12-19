
import requests
from bs4 import BeautifulSoup

url = "https://www.imss.gob.mx/personas-trabajadoras-independientes/calculadora"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

try:
    response = requests.get(url, headers=headers, timeout=10)
    print(f"Status Code: {response.status_code}")
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Remove script, style, etc.
    for script in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]):
        script.decompose()
    
    text = soup.get_text(separator='\n')
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    
    print("--- Content Start ---")
    print(text[:1000])
    print("--- Content End ---")
    
except Exception as e:
    print(f"Error: {e}")
