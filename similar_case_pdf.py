"""
Avukatın serbest metinle anlattığı bir dava olayına en çok benzeyen kararları bulur ve
tam metinleriyle birlikte bir PDF raporu üretir.

Kullanım:
    python similar_case_pdf.py "Müvekkilim ile ortağı arasında..." --limit 5 --output rapor.pdf
    python similar_case_pdf.py --file dava_anlatimi.txt --output rapor.pdf
"""

import argparse
import os
import sys
from datetime import datetime
from xml.sax.saxutils import escape

import requests

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak

from retriever import find_similar_decisions, get_section

# Burada LLM'i tool-calling icin degil, tek seferlik duz metin uretimi icin kullaniyoruz.
# query.py'deki agentic akistaki guvenilirlik sorunlari (tool_choice'un yok sayilmasi,
# yanlis parametre uretimi vb.) hep modelin "hangi arac cagrilacak" kararini vermesinden
# kaynaklaniyordu. Burada boyle bir karar yok - sadece verilen ozet metne dayanarak
# aciklama yazmasi isteniyor, bu yuzden cok daha guvenilir calisir.
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.1:8b"

_JUSTIFY_SYSTEM = (
    "Sen bir Turk hukuku asistanisin. Sana bir avukatin anlattigi olay ile veri "
    "tabanindaki bir mahkeme kararinin dava dilekcesi ve hukmu verilecek. Gorevin: bu "
    "kararin anlatilan olaya hukuki acidan NEDEN benzedigini 3-4 cumleyle Turkce acikla.\n\n"
    "ODAKLANMAN GEREKENLER:\n"
    "- Hukuki iliskinin turu ve dayanagi (orn. ortaklıktan cikma/cikarma, haklı sebep, "
    "TTK/TBK maddeleri, talep edilen hukuki sonuc)\n"
    "- Taraflarin hukuki konumu ve uyusmazligin ozu (kim, kimden, neden, ne talep ediyor)\n"
    "- Kararda verilen hukuki sonucun anlatilan olaydaki talep ile ne olcude ortustugu\n\n"
    "KESINLIKLE YAPMA: Sadece ortak kelime kullanimindan bahsetme ('ortaklik', 'sirket' gibi "
    "kelimeler her iki metinde de geciyor demek yetmez). Yuzeysel kelime benzerligi degil, "
    "hukuki iliskinin ozdesligini veya farkliligini analiz et. Sadece sana verilen bilgilere "
    "dayan, uydurma detay ekleme. Eger hukuki acidan zayif bir benzerlik varsa bunu acikca belirt."
)


def _llm_justification(case_text: str, karar: dict, dava_excerpt: str = "") -> str:
    """Verilen karar ile anlatilan olayin neden benzedigini LLM'e tek seferlik, duz bir
    tamamlama istegiyle acikliyoruz (tool cagirma yok, bu yuzden guvenilirligi yuksek).
    Not: eslesen anahtar kelimeleri BILEREK prompta vermiyoruz - model bunlari gorunce
    hukuki analiz yerine kelime eslestirmesine kayiyordu (gozlemlenen davranis)."""
    prompt = (
        f"ANLATILAN OLAY:\n{case_text.strip()}\n\n"
        f"KARARIN DAVA TURU: {karar.get('dava_turu', '')}\n\n"
        f"KARARIN DAVA DILEKCESI (ozet):\n{dava_excerpt[:2000] or '(mevcut degil)'}\n\n"
        f"KARARIN HUKMU:\n{karar.get('huküm_ozet', '')}\n\n"
        f"Bu karar anlatilan olaya hukuki acidan neden benziyor (veya benzemiyor)? Kisa ve net acikla."
    )
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": _JUSTIFY_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "").strip()
        return content or "(LLM boş yanıt döndürdü)"
    except Exception as e:
        return f"(LLM değerlendirmesi alınamadı: {e})"

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Türkçe karakterler (ı, ğ, ş, ü, ö, ç) reportlab'ın varsayılan fontlarında (Helvetica vb.)
# doğru basılmıyor. Windows'ta hazır bulunan DejaVuSans TTF'i unicode desteği için kaydediyoruz.
_FONT_DIR = r"C:\Windows\Fonts"
_FONT_REGULAR = os.path.join(_FONT_DIR, "DejaVuSans.ttf")
_FONT_BOLD = os.path.join(_FONT_DIR, "DejaVuSans-Bold.ttf")

if os.path.exists(_FONT_REGULAR) and os.path.exists(_FONT_BOLD):
    pdfmetrics.registerFont(TTFont("Turkce", _FONT_REGULAR))
    pdfmetrics.registerFont(TTFont("Turkce-Bold", _FONT_BOLD))
    FONT_NAME = "Turkce"
    FONT_BOLD = "Turkce-Bold"
else:
    # Fallback: Turkce karakterler bozuk gorunebilir ama script en azindan calisir.
    print("UYARI: DejaVuSans fontu bulunamadi, Turkce karakterler duzgun gorunmeyebilir.")
    FONT_NAME = "Helvetica"
    FONT_BOLD = "Helvetica-Bold"

SECTION_LABELS = {
    "dava":      "DAVA",
    "cevap":     "CEVAP",
    "kanitlar":  "KANITLAR / DELİLLER",
    "gerekce":   "GEREKÇE",
    "delil_deg": "DELİLLERİN DEĞERLENDİRİLMESİ",
    "huküm":     "HÜKÜM",
}
# Okuyucu için mantıklı bir sıra - section dict'inin kendi sırası dosyadaki konum sırası
# olduğu için genelde zaten doğru, ama garanti altına almak için burada sabitliyoruz.
SECTION_ORDER = ["dava", "cevap", "kanitlar", "gerekce", "delil_deg", "huküm"]


def _styles():
    return {
        "title": ParagraphStyle("title", fontName=FONT_BOLD, fontSize=18, leading=22, spaceAfter=14),
        "meta_label": ParagraphStyle("meta_label", fontName=FONT_NAME, fontSize=10, leading=13, textColor="#555555"),
        "case_title": ParagraphStyle("case_title", fontName=FONT_BOLD, fontSize=14, leading=18, spaceBefore=6, spaceAfter=8),
        "meta": ParagraphStyle("meta", fontName=FONT_NAME, fontSize=10, leading=14, spaceAfter=10),
        "section_header": ParagraphStyle("section_header", fontName=FONT_BOLD, fontSize=11, leading=14,
                                          spaceBefore=10, spaceAfter=4, textColor="#1a3a6b"),
        "body": ParagraphStyle("body", fontName=FONT_NAME, fontSize=9.5, leading=14, spaceAfter=6,
                                alignment=4),  # 4 = justify
    }


def _p(text: str, style: ParagraphStyle) -> Paragraph:
    safe = escape(text).replace("\n", "<br/>")
    return Paragraph(safe, style)


def build_pdf(case_text: str, output_path: str, limit: int = 5, with_llm: bool = True) -> None:
    result = find_similar_decisions(case_text, limit=limit)
    styles = _styles()
    story = []

    story.append(_p("Benzer Davalar Raporu", styles["title"]))
    story.append(_p(f"Oluşturulma tarihi: {datetime.now().strftime('%d.%m.%Y %H:%M')}", styles["meta_label"]))
    story.append(_p(f"Kullanılan anahtar kelimeler: {', '.join(result.get('anahtar_kelimeler', []))}",
                     styles["meta_label"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(_p("Anlatılan olay:", styles["section_header"]))
    story.append(_p(case_text.strip(), styles["body"]))
    story.append(Spacer(1, 0.6 * cm))

    kararlar = result.get("kararlar", [])
    if not kararlar:
        story.append(_p("Bu olaya benzer bir karar veri tabanında bulunamadı.", styles["body"]))
    else:
        for i, karar in enumerate(kararlar, start=1):
            if i > 1:
                story.append(PageBreak())

            story.append(_p(f"{i}. Benzer Karar — {karar['dava_turu'] or 'Belirtilmemiş'}",
                             styles["case_title"]))
            meta = (
                f"Mahkeme: {karar.get('mahkeme', '')}<br/>"
                f"Esas No: {karar.get('esas_no', '')} &nbsp;&nbsp; "
                f"Karar No: {karar.get('karar_no', '')}<br/>"
                f"Karar Tarihi: {karar.get('karar_tarihi', '')}<br/>"
                f"Eşleşen anahtar kelimeler: {', '.join(karar.get('eslesen_kelimeler', []))}"
            )
            story.append(Paragraph(meta, styles["meta"]))

            # Bölüm metinlerini önce tek seferde çekiyoruz - hem LLM gerekçesinde
            # (dava dilekçesi özeti olarak) hem de PDF'e tam metin basarken kullanılacak.
            doc_id = karar["doc_id"]
            section_texts = {}
            for section_key in SECTION_ORDER:
                if section_key not in karar.get("sections", []):
                    continue
                section_result = get_section(doc_id, section_key)
                if "hata" not in section_result:
                    section_texts[section_key] = section_result["content"]

            if with_llm:
                print(f"  [{i}/{len(kararlar)}] LLM'den benzerlik gerekçesi isteniyor "
                      f"({doc_id})...")
                dava_excerpt = section_texts.get("dava", "")
                justification = _llm_justification(case_text, karar, dava_excerpt)
                story.append(_p("Neden Benzer? (LLM Değerlendirmesi)", styles["section_header"]))
                story.append(_p(justification, styles["body"]))

            for section_key in SECTION_ORDER:
                if section_key not in section_texts:
                    continue
                story.append(_p(SECTION_LABELS.get(section_key, section_key.upper()),
                                 styles["section_header"]))
                story.append(_p(section_texts[section_key], styles["body"]))

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
    )
    doc.build(story)


def main():
    parser = argparse.ArgumentParser(description="Benzer dava kararlarını bulup PDF rapor üretir.")
    parser.add_argument("case_text", nargs="?", help="Dava anlatımı (doğrudan metin olarak)")
    parser.add_argument("--file", help="Dava anlatımının bulunduğu metin dosyası")
    parser.add_argument("--limit", type=int, default=5, help="Kaç benzer karar getirilecek (varsayılan 5)")
    parser.add_argument("--output", default="benzer_davalar_raporu.pdf", help="Çıktı PDF dosya yolu")
    parser.add_argument("--no-llm", action="store_true",
                         help="LLM benzerlik gerekçesini atla (daha hızlı, sadece kod skorlaması)")
    args = parser.parse_args()

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            case_text = f.read()
    elif args.case_text:
        case_text = args.case_text
    else:
        print("Dava anlatımını doğrudan argüman olarak ya da --file ile verin.")
        sys.exit(1)

    build_pdf(case_text, args.output, limit=args.limit, with_llm=not args.no_llm)
    print(f"Rapor oluşturuldu -> {args.output}")


if __name__ == "__main__":
    main()
