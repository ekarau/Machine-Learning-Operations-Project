# tests/test_smoke.py
import requests
import time
import sys

# Konfigürasyon
API_URL = "http://127.0.0.1:8000"
MAX_RETRIES = 5
WAIT_SECONDS = 5

def wait_for_service():
    """Servisin ayağa kalkmasını bekler."""
    print(f"Servise bağlanmaya çalışılıyor: {API_URL}/health")
    for i in range(MAX_RETRIES):
        try:
            response = requests.get(f"{API_URL}/health")
            if response.status_code == 200:
                print("Servis ayakta!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        
        print(f"Bekleniyor... ({i+1}/{MAX_RETRIES})")
        time.sleep(WAIT_SECONDS)
    return False

def run_smoke_test():
    """Gerçek bir tahmin isteği gönderir (Smoke Test)."""
    
    # 1. Önce servisin hazır olmasını bekle
    if not wait_for_service():
        print("HATA: Servis belirtilen sürede ayağa kalkmadı.")
        sys.exit(1)

    # 2. Tahmin isteği gönder (Payload ödev senaryosuna uygun olmalı)
    payload = {
        "Student_ID": "SMOKE_TEST_USER",
        "Age": 25,
        "Progress_Percentage": 50.0,
        "Quiz_Score_Avg": 80.0
    }
    
    try:
        print("Tahmin isteği gönderiliyor...")
        response = requests.post(f"{API_URL}/predict", json=payload)
        
        # 3. Sonucu kontrol et
        if response.status_code == 200:
            print("BAŞARILI: Smoke Test geçti. API 200 OK döndü.")
            print("Cevap:", response.json())
            sys.exit(0) # Başarılı çıkış kodu
        else:
            print(f"BAŞARISIZ: API {response.status_code} döndü.")
            print("Detay:", response.text)
            sys.exit(1) # Hata çıkış kodu
            
    except Exception as e:
        print(f"BAŞARISIZ: İstek sırasında hata oluştu: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_smoke_test()
