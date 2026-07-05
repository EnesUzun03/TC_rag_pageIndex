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

# LLM bazen Türkçe kelimeleri aksansız yazıyor ("ortakliktan cikma" vs "Ortaklıktan Çıkma").
# str.lower() bu karakterleri eşitlemediği için ("ı" != "i", "ç" != "c") aramalar sessizce 0 sonuç
# döndürebiliyor. Karşılaştırma öncesi hem sorguyu hem metni bu tabloyla sadeleştiriyoruz.
_TR_FOLD = str.maketrans({
    "ı": "i", "İ": "i", "I": "i",
    "ğ": "g", "Ğ": "g",
    "ü": "u", "Ü": "u",
    "ş": "s", "Ş": "s",
    "ö": "o", "Ö": "o",
    "ç": "c", "Ç": "c",
})


def _fold(text: str) -> str:
    return text.lower().translate(_TR_FOLD)

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
    limit = int(limit)  # LLM bazen sayıyı string olarak gönderiyor ("15")
    # dava_turu gerçekten index'te var mı kontrol et; yoksa yoksay
    dava_turu_f = _fold(dava_turu)
    if dava_turu:
        turu_var = any(dava_turu_f in _fold(k) for k in idx["by_dava_turu"])
        if not turu_var:
            dava_turu = ""  # index'te olmayan bir tür → keyword-only moda geç
            dava_turu_f = ""

    keyword_f = _fold(keyword)
    results = []
    for doc in idx["docs"]:
        dt = doc.get("dava_turu", "") # dava türlerini çek
        huküm = doc.get("huküm_ozet", "")
        ozet = " ".join(doc.get("section_ozetleri", {}).values())
        dt_f = _fold(dt)

        #Eğer dava türü verilmişse kontrol et.
        turu_esles = dava_turu_f in dt_f if dava_turu else True
        if not turu_esles:
            continue  # dava türü uymuyorsa devam etmeye gerek yok, tam metin okumaktan kaçınırız

        # Keyword'ün nerede eşleştiğine göre alaka skoru veriyoruz: dava_turu alanında geçmesi
        # en güçlü sinyal (davanın konusunu doğrudan tanımlıyor), sonra hüküm, sonra özetler,
        # en zayıfı da sadece tam metinde geçiyor olması. Hüküm metnindeki kelime tek başına
        # güvenilir değil - örn. "tazminat" kelimesi boilerplate ("kötü niyet tazminatı") olarak
        # konuyla ilgisiz kararlarda da sıkça geçiyor, bu yüzden dava_turu eşleşmesi önceliğe alındı.
        keyword_esles = True
        relevans = 100 if (dava_turu and dava_turu_f == dt_f) else 0
        if keyword:
            if keyword_f in dt_f:
                relevans += 60
                keyword_esles = True
            elif keyword_f in _fold(huküm):
                relevans += 50
                keyword_esles = True
            elif keyword_f in _fold(ozet):
                relevans += 20
                keyword_esles = True
            else:
                path = os.path.join(KARARLAR_DIR, doc["dosya"])
                try:
                    with open(path, encoding="utf-8") as f:
                        full_text = f.read()
                    keyword_esles = keyword_f in _fold(full_text)
                    if keyword_esles:
                        relevans += 5
                except Exception:
                    keyword_esles = False
        else:
            relevans += 10  # keyword yok, sadece dava_turu filtresiyle geldi

        if turu_esles and keyword_esles: # her ikisinde sağlıyorsa sonuçlara ekle
            results.append({
                "doc_id":       doc["doc_id"],
                "mahkeme":      doc.get("mahkeme", ""),
                "dava_turu":    dt,
                "karar_tarihi": doc.get("karar_tarihi", ""),
                "esas_no":      doc.get("esas_no", ""),
                "karar_no":     doc.get("karar_no", ""),
                "sections":     doc.get("sections", []),
                "huküm_ozet":   huküm[:200],
                "_relevans":    relevans,
            })

    # En alakalı sonuçlar başta olacak şekilde sırala, sonra limitle kes
    results.sort(key=lambda r: r["_relevans"], reverse=True)
    results = results[:limit]
    for r in results:
        del r["_relevans"]  # LLM'e gereksiz iç detay gitmesin

    if not results:
        return {
            "toplam_eslesen": 0,
            "kararlar": [],
            "uyari": (
                "Veri tabanında bu sorguya uygun karar bulunamadi. "
                "Bu veri tabani yalnizca Asliye Ticaret Mahkemesi kararlarini icermektedir. "
                "Aile hukuku, ceza veya idare davalarına ait karar bulunmamaktadir. "
                "Kullaniciya nazikce bilgi ver, uydurmа yanıt verme."
            ),
        }
    return {"toplam_eslesen": len(results), "kararlar": results}


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
