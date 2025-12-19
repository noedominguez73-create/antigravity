import requests
import time

# Configuration
BASE_URL = "http://localhost:5000"
PROFESSIONAL_ID = 1 # Change to a valid ID

def test_rate_limit():
    print("[INFO] Iniciando Test de Rate Limiting (10 req/min)...")
    
    url = f"{BASE_URL}/api/chatbot/{PROFESSIONAL_ID}/chat"
    headers = {"Content-Type": "application/json"}
    
    # Send 12 requests
    for i in range(1, 13):
        data = {
            "message": f"Test message {i}",
            "session_id": "test_session_limit"
        }
        
        try:
            start = time.time()
            response = requests.post(url, json=data, headers=headers)
            elapsed = time.time() - start
            
            if response.status_code == 200:
                print(f"[OK] Req {i}: OK ({elapsed:.2f}s)")
            elif response.status_code == 429:
                print(f"[BLOCKED] Req {i}: BLOQUEADO (429 Too Many Requests) - Rate Limit Funciona!")
                break # Success
            else:
                print(f"[INFO] Req {i}: Status {response.status_code} - {response.text[:50]}")
                
        except Exception as e:
            print(f"[ERROR] Error de conexion: {e}")
            print("Asegurate de que el servidor este corriendo.")
            break

if __name__ == "__main__":
    test_rate_limit()
