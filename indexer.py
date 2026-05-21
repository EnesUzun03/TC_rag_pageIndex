import os
import re
import json

KARARLAR_DIR = "kararlar"
INDEX_FILE = "kararlar_index.json"

#Regex listesi , birden fazla regular expression (düzenli ifade) deseninin bir arada tutulduğu listedir.Her eleman ("isim", "regex")
#Mahkeme kararındaki bölümleri bulmak için regex listesi 
SECTION_PATTERNS = [
    ("dava",       r"(?:^|\n)\s*DAVA\s*:\s*(?!TARİHİ)"),
    ("cevap",      r"(?:^|\n)\s*CEVAP\s*:\s*"),
    ("kanitlar",   r"(?:^|\n)\s*(?:KANITLAR|DELİLLER)\s*:\s*"),
    ("gerekce",    r"(?:^|\n)\s*GEREKÇE\s*:\s*"),
    ("delil_deg",  r"(?:^|\n)\s*DELİLLERİN DEĞERLENDİRİLMESİ(?:\s*VE GEREKÇE)?\s*[:\;]?\s*"),
    ("huküm",      r"(?:^|\n)\s*H\s*[UÜ]\s*K\s*[UÜ]\s*M\s*[:\;]?\s*"),
]
#Kararın üst kısmındaki metadata bilgilerini çıkarmak için regex listesi
HEADER_PATTERNS = {
    "esas_no":      r"ESAS NO\s*[:\t]+\s*(.+)",
    "karar_no":     r"KARAR NO\s*[:\t]+\s*(.+)",
    "dava_turu":    r"DAVA\s*[:\t]+\s*(?!TARİHİ)(.+)",
    "dava_tarihi":  r"DAVA TARİHİ\s*[:\t]+\s*(.+)",
    "karar_tarihi": r"KARAR TARİHİ\s*[:\t]+\s*(.+)",
}


def parse_header(text: str) -> dict:
    lines = text[:800] # Header genelde ilk kısım olduğu için 800 karakter ile sınırladık
    meta = {}
    for key, pattern in HEADER_PATTERNS.items():
        m = re.search(pattern, lines) 
        meta[key] = m.group(1).strip() if m else "" # Burada eşleşme varsa kaydeder bulamazsa da boş string verir

    # Mahkeme: ilk 4 satır arasında T.C. sonrası iki satır
    header_lines = [l.strip() for l in text.splitlines()[:6] if l.strip()]
    mahkeme_parts = []
    for i, line in enumerate(header_lines):
        if line == "T.C.":
            mahkeme_parts = header_lines[i + 1: i + 3] # T.C. sonrası iki satır
            break
    meta["mahkeme"] = " ".join(mahkeme_parts) # Birleştiridik ve tek bir string yaptık
    return meta

# Kararın içindeki bölümleri bulur.
def find_sections(text: str) -> dict:
    hits = []
    for name, pattern in SECTION_PATTERNS: # Tüm section regexlerini dolaşır. 
        for m in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
            hits.append((m.start(), name)) # Eşleşen bölümlerin başlangıç pozisyonu ve bölüm adını kaydeder. Böylece kararın hangi karakterinden itibaren hangi bölümün başladığını biliriz.

    hits.sort(key=lambda x: x[0]) # bölümleri metindeki sırasına göre dizdik
    # Aynı section birden fazla kez bulunursa ilkini tutacak
    seen = set()
    unique = []
    for pos, name in hits: 
        if name not in seen: # görülenler içierisinde yoksa unique listesine ekle ve seen setine kaydet
            unique.append((pos, name))
            seen.add(name)

    sections = {}
    for i, (start, name) in enumerate(unique):
        end = unique[i + 1][0] if i + 1 < len(unique) else len(text) # Son bölümün sonu metnin sonu olur
        content = text[start:end].strip() # her bölümün içeriğini alır ve baştaki ve sondaki boşlukları temizler
        sections[name] = { # sözlüğe bölüm adını, başlangıç ve bitiş karakter pozisyonlarını ve bölümün ilk 400 karakterinden oluşan özetini kaydeder
            "char_start": start,
            "char_end": end,
            "ozet": content[:400].replace("\n", " "),
        }
    return sections

# Ana işlem metodu burası
def build_index():
    index = {"docs": [], "by_dava_turu": {}, "by_mahkeme": {}} #indexler 3'e ayrılır. tüm kararların listelendiği "docs", dava türüne göre gruplanmış "by_dava_turu" ve mahkemeye göre gruplanmış "by_mahkeme"

    files = sorted(f for f in os.listdir(KARARLAR_DIR) if f.endswith(".txt")) #Tüm txt dosyalarını alırız
    print(f"{len(files)} dosya bulundu, indexleniyor...")

    for fname in files: # Her dosya için 
        path = os.path.join(KARARLAR_DIR, fname) # dosya yolunu oluşturur
        with open(path, encoding="utf-8") as f: # dosyayı açar ve içeriğini okuruz
            text = f.read()

        doc_id = fname.replace(".txt", "") # dosya adını ıd olarak aldık
        meta = parse_header(text) # metadataları çıkardık
        sections = find_sections(text) # bölümleri bulduk

        # HÜKÜM özetini daha temiz çek (son bölüm genelde kısa ve net)
        huküm_text = ""
        if "huküm" in sections:
            raw = text[sections["huküm"]["char_start"]: sections["huküm"]["char_end"]]
            huküm_text = raw.strip()[:600].replace("\n", " ")

        # Tek bir dava için tüm bilgileri birleştirerek bir sözlük oluşturduk ve indexe ekledik
        doc = {
            "doc_id": doc_id,
            "dosya": fname,
            **meta,
            "sections": list(sections.keys()), # bölüm isimleri
            "section_ozetleri": {k: v["ozet"] for k, v in sections.items()},
            "huküm_ozet": huküm_text,
            "section_offsets": {k: {"start": v["char_start"], "end": v["char_end"]} for k, v in sections.items()}, # section offset bilgileri
        }
        index["docs"].append(doc)

        # by_dava_turu
        dt = meta.get("dava_turu", "Diğer").strip() or "Diğer"
        index["by_dava_turu"].setdefault(dt, []).append(doc_id)

        # by_mahkeme
        mk = meta.get("mahkeme", "Bilinmiyor").strip() or "Bilinmiyor"
        index["by_mahkeme"].setdefault(mk, []).append(doc_id)

        bolumler = ", ".join(sections.keys())
        dava_t = meta.get('dava_turu', '?').encode('ascii', 'replace').decode()
        print(f"  OK {doc_id} | {dava_t} | {bolumler}")

    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"\nIndex kaydedildi -> {INDEX_FILE}")
    print(f"Toplam dava turu sayisi: {len(index['by_dava_turu'])}")


if __name__ == "__main__":
    build_index()
