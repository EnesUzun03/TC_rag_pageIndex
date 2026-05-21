import requests
import json
import time
import sys
from bs4 import BeautifulSoup
import os

sys.stdout.reconfigure(encoding='utf-8')

# Kararların kaydedileceği klasör
OUTPUT_DIR = "kararlar"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def get_decision_list(keyword="boşanma", page_number=1, page_size=10):
    url = "https://emsal.uyap.gov.tr/aramalist"
    
    payload = {
        "data": {
            "aranan": keyword,
            "arananKelime": keyword,
            "pageSize": page_size,
            "pageNumber": page_number
        }
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json"
    }
    
    for attempt in range(5):
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            print(f"Uyarı: 429 Too Many Requests (Liste). {10 * (attempt+1)} saniye bekleniyor...")
            time.sleep(10 * (attempt + 1))
        else:
            print(f"Liste çekilemedi. Status code: {response.status_code}")
            return None
            
    print("Maksimum deneme sayısına ulaşıldı (Liste).")
    return None

def get_decision_content(decision_id):
    url = f"https://emsal.uyap.gov.tr/getDokuman?id={decision_id}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    for attempt in range(5):
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if "data" in data and data["data"]:
                html_content = data["data"]
                # HTML etiketlerini temizle
                soup = BeautifulSoup(html_content, "html.parser")
                return soup.get_text(separator="\n", strip=True)
        elif response.status_code == 429:
            print(f"Uyarı: 429 Too Many Requests (İçerik). {10 * (attempt+1)} saniye bekleniyor...")
            time.sleep(10 * (attempt + 1))
        else:
            print(f"İçerik çekilemedi (ID: {decision_id}). Status code: {response.status_code}")
            return None
            
    print(f"Maksimum deneme sayısına ulaşıldı (İçerik: {decision_id}).")
    return None

def main():
    keyword = "boşanma"
    page_size = 100
    
    print(f"'{keyword}' konulu kararlar aranıyor...")
    
    # Toplam karar sayısını öğrenmek için ilk isteği atıyoruz
    initial_response = get_decision_list(keyword, 1, page_size)
    
    if not initial_response or "data" not in initial_response or "data" not in initial_response["data"]:
        print("Karar listesi alınamadı.")
        return
        
    total_records = initial_response["data"].get("recordsTotal", 0)
    if total_records == 0:
        print("Hiç karar bulunamadı.")
        return
        
    import math
    total_pages = math.ceil(total_records / page_size)
    print(f"Toplam {total_records} karar bulundu. Her sayfada {page_size} karar olacak şekilde {total_pages} sayfa indirilecek...")
    
    for page_number in range(1, total_pages + 1):
        print(f"\n--- {page_number}. Sayfa İndiriliyor ---")
        list_response = get_decision_list(keyword, page_number, page_size)
        
        if not list_response or "data" not in list_response or "data" not in list_response["data"]:
            print(f"{page_number}. sayfa alınamadı, atlanıyor.")
            continue
            
        decisions = list_response["data"]["data"]
        
        for decision in decisions:
            decision_id = decision.get("id")
            esas_no = decision.get("esasNo", "esas_yok").replace("/", "_")
            karar_no = decision.get("kararNo", "karar_yok").replace("/", "_")
            
            filename = os.path.join(OUTPUT_DIR, f"{esas_no}_{karar_no}.txt")
            
            # Eğer dosya zaten varsa tekrar indirmemek için kontrol
            if os.path.exists(filename):
                print(f"[{esas_no} / {karar_no}] Zaten indirilmiş, atlanıyor...")
                continue
                
            print(f"[{esas_no} / {karar_no}] Karar indiriliyor... ID: {decision_id}")
            
            content = get_decision_content(decision_id)
            
            if content:
                # Kararı metin dosyasına kaydet
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"--> {filename} kaydedildi.")
            
            # Sunucuyu yormamak için kısa bir bekleme süresi
            time.sleep(1.5)

if __name__ == "__main__":
    main()
