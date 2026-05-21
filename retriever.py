"""
PageIndex yaklaşımıyla LLM'e sunulan 3 araç.
LLM önce master index'i görür → ilgili kararı seçer → bölüm yapısını inceler → sadece
ihtiyaç duyduğu bölümü getirir.
"""

import json
import os

# Bu json içerisinde section özetleri isimleri ve offsetleri var 
INDEX_FILE = "kararlar_index.json"
KARARLAR_DIR = "kararlar"

_index_cache = None

# Burada json'u ram'e yükleyip cache'liyoruz, böylece her araç çağrısında diske gitmek zorunda kalmayız.
def _load_index() -> dict:
    global _index_cache
    if _index_cache is None:
        with open(INDEX_FILE, encoding="utf-8") as f:
            _index_cache = json.load(f)
    return _index_cache


# ── Araç 1 ────────────────────────────────────────────────────────────────────
# LLm'in ilk çağırması gereken araç, böylece hangi dava türlerinin mevcut olduğunu görür ve sonraki aramalarda bu türleri kullanabilir.
def get_master_index() -> dict:
    """
    Veri tabanı istatistiklerini ve dava türü listesini döndürür.
    Hangi dava türlerinin mevcut olduğunu görmek için önce bunu çağır.
    Belge listesi için search_decisions kullan.
    """
    idx = _load_index()
    dava_turu_counts = {k: len(v) for k, v in idx["by_dava_turu"].items()} # Her dava türünde kaç karar var? k dava türü v sayısı {"Tazminat": 120, "Boşanma": 80, ...}
    return {
        "toplam_karar": len(idx["docs"]),
        "dava_turu_sayisi": len(idx["by_dava_turu"]),
        "dava_turu_ve_adet": dava_turu_counts,
    }


# ── Araç 2 ────────────────────────────────────────────────────────────────────
# LLm'in belirli bir dava türünde veya anahtar kelimeyle karar araması yapmasını sağlar. get_master_index'ten öğrendiği dava türlerini kullanarak arama yapabilir.
def search_decisions(dava_turu: str = "", keyword: str = "", limit: int = 15) -> dict:
    """
    Dava türü veya anahtar kelimeye göre karar listesi döndürür (max 15 sonuç).
    """
    idx = _load_index()
    # dava_turu gerçekten index'te var mı kontrol et; yoksa yoksay
    if dava_turu:
        turu_var = any(dava_turu.lower() in k.lower() for k in idx["by_dava_turu"])
        if not turu_var:
            dava_turu = ""  # index'te olmayan bir tür → keyword-only moda geç

    results = []
    for doc in idx["docs"]:
        dt = doc.get("dava_turu", "") # dava türlerini çek
        huküm = doc.get("huküm_ozet", "")
        ozet = " ".join(doc.get("section_ozetleri", {}).values())

        #Eğer dava türü verilmişse kontrol et.
        turu_esles = dava_turu.lower() in dt.lower() if dava_turu else True

        if keyword and not keyword.lower() in (dt + huküm + ozet).lower(): # Önce dava türü, hüküm özeti ve bölüm özetlerinde ara
            # Özette bulunamazsa tam metni ara
            path = os.path.join(KARARLAR_DIR, doc["dosya"])
            try:
                with open(path, encoding="utf-8") as f:
                    full_text = f.read()
                keyword_esles = keyword.lower() in full_text.lower() # Full arama yap
            except Exception:
                keyword_esles = False
        elif keyword:
            keyword_esles = True
        else:
            keyword_esles = True

        if turu_esles and keyword_esles: # her ikisinde sağlıyorsa sonuçlara ekle
            results.append({
                "doc_id":       doc["doc_id"],
                "mahkeme":      doc.get("mahkeme", ""),
                "dava_turu":    dt,
                "karar_tarihi": doc.get("karar_tarihi", ""),
                "esas_no":      doc.get("esas_no", ""),
                "sections":     doc.get("sections", []),
                "huküm_ozet":   huküm[:200],
            })
        if len(results) >= limit:
            break

    return {"toplam_eslesen": len(results), "kararlar": results} # Tüm eşleşen kararlar sonuç olarak döndürülür.


# ── Araç 3 ────────────────────────────────────────────────────────────────────
# Kararların hepsini vermeden önce hangi bölümlerin olduğunu ve her bölümün ne hakkında olduğunu görmek için bu aracı kullanır. Böylece sadece ihtiyaç duyduğu bölümü çağırır.
def get_decision_structure(doc_id: str) -> dict:
    """
    Belirli bir kararın bölüm yapısını ve her bölümün kısa özetini döndürür.
    LLM hangi bölümü getireceğine bu araçla karar verir.
    """
    idx = _load_index()
    doc = next((d for d in idx["docs"] if d["doc_id"] == doc_id), None) # ilk eşleşen kararı bul
    if doc is None:
        return {"hata": f"'{doc_id}' bulunamadı."}

    return {
        "doc_id":          doc_id,
        "mahkeme":         doc.get("mahkeme", ""),
        "dava_turu":       doc.get("dava_turu", ""),
        "esas_no":         doc.get("esas_no", ""),
        "karar_no":        doc.get("karar_no", ""),
        "dava_tarihi":     doc.get("dava_tarihi", ""),
        "karar_tarihi":    doc.get("karar_tarihi", ""),
        "mevcut_bolumler": doc.get("sections", []),
        "bolum_ozetleri":  doc.get("section_ozetleri", {}), # bölüm özetleri burada llm bu özetlere bakarak hangi bölümün ihtiyaç duyduğunu anlayabilir
    }


# ── Araç 3 ────────────────────────────────────────────────────────────────────
# LLm'in gerçek ihtiyacını burası getirir. 
def get_section(doc_id: str, section: str) -> dict:
    """
    Belirli bir kararın istenen bölümünün tam metnini döndürür.
    section: 'dava' | 'cevap' | 'kanitlar' | 'gerekce' | 'delil_deg' | 'huküm'
    """
    idx = _load_index()
    doc = next((d for d in idx["docs"] if d["doc_id"] == doc_id), None) # ilk eşleşen kararı bul
    if doc is None:
        return {"hata": f"'{doc_id}' bulunamadı."}

    offsets = doc.get("section_offsets", {}) # Bölümün offset bilgilerini alıyorum
    if section not in offsets:
        available = list(offsets.keys())
        return {"hata": f"'{section}' bölümü yok. Mevcut: {available}"}

    path = os.path.join(KARARLAR_DIR, doc["dosya"])
    with open(path, encoding="utf-8") as f:
        text = f.read() # Gerçek veriyi okuyoruz

    start = offsets[section]["start"]
    end   = offsets[section]["end"]
    content = text[start:end].strip() # Offset bilgisine göre bölümü kesip getiriyoruz

    return {
        "doc_id":  doc_id,
        "section": section,
        "esas_no": doc.get("esas_no", ""),
        "content": content,
    }


# ── Tool schema (LLM için şema)  ─────────────────────────────────────
TOOLS = [
    {
        "name": "get_master_index",
        "description": (
            "Tüm hukuk kararlarının üst düzey listesini ve özetlerini getirir. "
            "Hangi kararın soruyla ilgili olduğunu bulmak için önce bu aracı çağır."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "search_decisions",
        "description": (
            "Dava türü veya anahtar kelimeye göre karar arar, max 15 sonuç döndürür. "
            "get_master_index'ten dava türünü öğrendikten sonra bu araçla ilgili kararları bul."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dava_turu": {
                    "type": "string",
                    "description": "Filtrelenecek dava türü (kısmi eşleşme), örn. 'Tazminat'",
                },
                "keyword": {
                    "type": "string",
                    "description": "Karar metinlerinde aranacak anahtar kelime, örn. 'zimmet'",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maksimum sonuç sayısı (varsayılan 15)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_decision_structure",
        "description": (
            "Belirli bir kararın (doc_id) bölüm yapısını ve her bölümün kısa özetini getirir. "
            "Hangi bölümde arama yapacağını belirlemek için bu aracı kullan."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "doc_id": {
                    "type": "string",
                    "description": "Kararın dosya adından türetilen ID, örn. '2009_131_2024_21'",
                }
            },
            "required": ["doc_id"],
        },
    },
    {
        "name": "get_section",
        "description": (
            "Kararın belirli bir bölümünün tam metnini getirir. "
            "Mümkün olduğunca dar tut — sadece ihtiyaç duyduğun bölümü çağır."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "doc_id": {
                    "type": "string",
                    "description": "Karar ID'si",
                },
                "section": {
                    "type": "string",
                    "enum": ["dava", "cevap", "kanitlar", "gerekce", "delil_deg", "huküm"],
                    "description": "Getirilecek bölüm adı",
                },
            },
            "required": ["doc_id", "section"],
        },
    },
]


def dispatch(tool_name: str, tool_input: dict):
    """Tool çağrısını çalıştırır, eksik argümanlarda hata mesajı döner."""
    try:
        if tool_name == "get_master_index":
            return get_master_index()
        if tool_name == "search_decisions":
            return search_decisions(**tool_input)
        if tool_name == "get_decision_structure":
            return get_decision_structure(**tool_input)
        if tool_name == "get_section":
            return get_section(**tool_input)
        return {"hata": f"Bilinmeyen arac: {tool_name}"}
    except TypeError as e:
        return {"hata": f"Eksik parametre: {e}. Lutfen gerekli alanlari doldurun."}
