
import requests
from bs4 import BeautifulSoup

url = "https://www.imss.gob.mx/personas-trabajadoras-independientes"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

if __name__ == "__main__":
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = response.apparent_encoding  # Fix encoding
        
        print(f"Status Code: {response.status_code}")
        
        soup = BeautifulSoup(response.text, 'html.parser') # Use .text not .content
        
        # Remove script, style, etc.
        for script in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]):
            script.decompose()
        
        # Better text extraction
        text = soup.get_text(separator=' ', strip=True)
        
        print("--- Content Start ---")
        print(text[:1000])
        print("--- Content End ---")
        
    except Exception as e:
        print(f"Error: {e}")
