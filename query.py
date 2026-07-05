"""
PageIndex tabanlı hukuk RAG — Ollama / llama3.1:8b sorgu arayüzü.
"""

import sys
import json
import requests

# Windows konsolunda UTF-8 zorla
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from retriever import TOOLS, dispatch # TOOLS LLM'e verilecek tool tanımları, dispatch ise çağrılan tool'ları çalıştıracak fonksiyon.

# Ollama local API endpointi ve model adı
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL      = "llama3.1:8b"
MAX_ROUNDS = 8

SYSTEM = """Sen bir Turk hukuku asistanisin. Elinde mahkeme kararlarindan olusan bir veri tabani var.

ZORUNLU SIRALAMA:
1. get_master_index() -> mevcut dava turlerini ve adedi gor.
2. Kullanicinin sorusundaki konuyla ilgili kelimeyi dava_turu_ve_adet listesinde ara. AYNEN eslesen bir deger varsa:
   search_decisions(dava_turu="<listedeki deger>") cagir. Eslesen deger yoksa: search_decisions(dava_turu="", keyword="<sorudaki konuyla ilgili gercek kelime>") cagir.
   Ornek kelimeleri (zimmet, bosanma vb.) asla dogrudan kullanma, sadece kullanicinin GERCEK sorusundaki kelimeleri kullan. dava_turu'nu uydurma.
3. Sonuclari goren doc_id'ler icin get_decision_structure() cagir.
4. get_section() ile sadece gerekli bolumu getir.
5. Kullaniciya Turkce ozet yanit ver.

KRITIK: dava_turu parametresine sadece get_master_index sonucunda GERCEKTEN GORUNEN bir degeri yaz. Yoksa bos birak."""

# Ollama tool schema biraz farklı — "parameters" yerine "input_schema" değil,
# OpenAI uyumlu format kullanıyor.
def _to_ollama_tools(tools: list) -> list:
    result = []
    for t in tools:
        result.append({
            "type": "function",
            "function": {
                "name":        t["name"],
                "description": t["description"],
                "parameters":  t["input_schema"],
            }
        })
    return result

# PageIndex zincirinin zorunlu sırası. Model bu sırayı atlayıp erken cevap veremesin diye
# her turda hangi tool'un çağrılabileceğini biz belirliyoruz (Ollama'nın tool_choice'una güvenmek
# yeterli değil - serbest bırakılınca model 2. turda hiç tool çağırmadan halüsinasyon üretiyordu).
def _required_tool(state: dict) -> str | None:
    if not state["master_done"]:
        return "get_master_index"
    if not state["search_done"]:
        return "search_decisions"
    if state["search_empty"]:
        return None  # eşleşme yok, modelin uyarıyı ilettiği serbest cevaba izin ver
    if not state["structure_done"]:
        return "get_decision_structure"
    if not state["section_done"]:
        return "get_section"
    return None  # zincir tamamlandı, artık serbest cevaba izin ver


# Kaynak/atıf bilgisini cevabın sonuna kod tarafında ekliyoruz - LLM'e "kaynak belirt" demek
# güvenilir değil (unutabilir/uydurabilir), bu yüzden hangi doc_id'den içerik okunduysa onun
# esas_no/karar_no/mahkeme bilgisini burada deterministik olarak ekliyoruz.
def _with_citation(answer: str, cited_doc_id: str | None, docs_meta: dict) -> str:
    if not cited_doc_id or cited_doc_id not in docs_meta:
        return answer
    meta = docs_meta[cited_doc_id]
    kaynak = (
        f"\n\nKaynak: {meta.get('mahkeme', '?')} | "
        f"Esas No: {meta.get('esas_no', '?')} | "
        f"Karar No: {meta.get('karar_no', '?')} | "
        f"Karar Tarihi: {meta.get('karar_tarihi', '?')}"
    )
    return answer + kaynak


# Bu sistemin llm agent sistemiyle nasıl çalıştığına dair temel akış:
def run_query(question: str, verbose: bool = True) -> str:
    messages = [
        {"role": "system",  "content": SYSTEM}, # LLM'e gönderilen Başlangıç context'i System prompt'u ve kullanıcı sorusu User mesajı olarak başlar.
        {"role": "user",    "content": question},
    ]
    ollama_tools = _to_ollama_tools(TOOLS) # LLM'e TOOL setini verdik

    state = {
        "master_done": False,
        "search_done": False,
        "search_empty": False,
        "structure_done": False,
        "section_done": False,
    }
    # Atıf için: search_decisions'ın döndürdüğü karar meta bilgileri (esas_no, mahkeme, vb.)
    # doc_id'ye göre saklanır. Cevap hangi doc_id'den üretildiyse (get_section'a bakılarak)
    # kaynağı buradan çekip cevabın sonuna kod tarafında ekleyeceğiz - LLM'e güvenmiyoruz.
    docs_meta = {}
    cited_doc_id = None
    top_doc_id = None  # search_decisions'ın seçtiği ana karar - get_section fallback'i için lazım

    # Ayni zorunlu tool ust uste kac kez basarisiz oldu - Ollama'nin tool_choice zorlamasi
    # bazen hicbir sekilde ise yaramiyor (model israrla baska bir tool cagiriyor), bu durumda
    # sonsuz donguye girmemek icin belirli bir esikten sonra adimi kendimiz calistiracagiz.
    stuck_tool = None
    stuck_count = 0

    # Bu döngü agent sistem döngüsü gibi düşünülebilir burada metodları çağırır ve sonuçları LLM'e geri veririz.
    for round_num in range(MAX_ROUNDS):
        required_tool = _required_tool(state)

        if required_tool == stuck_tool:
            stuck_count += 1
        else:
            stuck_tool = required_tool
            stuck_count = 0

        # get_section'da 2 turdur takılı kaldıysak, modele sormayı bırakıp kendimiz
        # varsayılan bir bölümle (tercihen "huküm") çağırıp zinciri kapatıyoruz.
        if required_tool == "get_section" and stuck_count >= 2 and top_doc_id:
            structure = docs_meta.get(top_doc_id, {})
            bolumler = structure.get("mevcut_bolumler", [])
            section = "huküm" if "huküm" in bolumler else (bolumler[0] if bolumler else None)
            if section:
                if verbose:
                    print(f"\n[Tur {round_num + 1}] !! GET_SECTION'DA TAKILDI, otomatik cagriliyor: "
                          f"get_section(doc_id={top_doc_id}, section={section})")
                section_result = dispatch("get_section", {"doc_id": top_doc_id, "section": section})
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "function": {
                            "name": "get_section",
                            "arguments": {"doc_id": top_doc_id, "section": section},
                        }
                    }],
                })
                messages.append({
                    "role": "tool",
                    "content": json.dumps(section_result, ensure_ascii=False),
                })
                if "hata" not in section_result:
                    state["section_done"] = True
                    cited_doc_id = section_result.get("doc_id")
                continue

        #Ollama API request body.
        payload = {
            "model":    MODEL,
            "messages": messages,
            "tools":    ollama_tools,
            "stream":   False,
        }
        if required_tool:
            payload["tool_choice"] = {
                "type": "function",
                "function": {"name": required_tool}
            }

        if verbose:
            print(f"\n[Tur {round_num + 1}] zorunlu_tool={required_tool or '(serbest)'}")

        resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()

        msg = data.get("message", {})
        tool_calls = msg.get("tool_calls", [])

        if verbose:
            print(f"  tool_calls={len(tool_calls)}")

        if not tool_calls:
            if required_tool:
                # Zincir tamamlanmadan modelin serbest cevap vermesine izin verme.
                # tool_choice'a rağmen tool çağırmadıysa, zorlayarak tekrar iste.
                # NOT: model küçük olduğu için burada verilecek talimat metnini bile
                # kelimesi kelimesine tool parametresi olarak kopyalayabiliyor. Bu yüzden
                # talimat yerine orijinal soruyu aynen tekrar ediyoruz.
                if verbose:
                    print(f"  !! ZORUNLU TOOL CAGRILMADI ({required_tool}), tekrar isteniyor")
                messages.append({"role": "assistant", "content": msg.get("content") or ""})
                messages.append({"role": "user", "content": question})
                continue
            return _with_citation(msg.get("content", ""), cited_doc_id, docs_meta)

        # Modelin birden fazla tool çağırmasını engelle: sadece ilk geçerliyi al.
        # Bir tool zorunluysa, sadece o isimdeki çağrıyı kabul et; diğerlerini yok say.
        valid_tc = None
        for tc in tool_calls:
            fn   = tc.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            if required_tool and name != required_tool:
                if verbose:
                    print(f"  !! ATLANDI (sirasi degil, beklenen={required_tool}): {name}")
                continue
            # Placeholder içeren çağrıları atla
            args_str = json.dumps(args)
            if "<doc_id>" in args_str or "<section>" in args_str:
                if verbose:
                    print(f"  !! ATLANDI (placeholder): {name}({args_str[:80]})")
                continue
            # Zorunlu parametre eksikse atla
            if name == "get_decision_structure" and not args.get("doc_id"):
                if verbose:
                    print(f"  !! ATLANDI (eksik doc_id): {name}")
                continue
            if name == "get_section" and (not args.get("doc_id") or not args.get("section")):
                if verbose:
                    print(f"  !! ATLANDI (eksik parametre): {name}")
                continue
            valid_tc = (tc, name, args)
            break  # Sadece ilk geçerli tool çağrısını işle

        if valid_tc is None:
            # Hiç geçerli tool yoksa modelden aynı tool'u tekrar dene (zorunluysa) ya da serbest cevap iste
            messages.append({"role": "assistant", "content": msg.get("content") or ""})
            if required_tool:
                messages.append({"role": "user", "content": question})
            else:
                messages.append({"role": "user", "content": "Elindeki bilgilerle Türkçe özet yanıt ver."})
            continue

        tc, name, args = valid_tc

        # Assistant mesajını geçmişe ekle (sadece geçerli 1 tool ile)
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": [tc]})

        if verbose:
            print(f"  -> {name}({json.dumps(args, ensure_ascii=True)[:100]})")

        result = dispatch(name, args)

        messages.append({
            "role":    "tool",
            "content": json.dumps(result, ensure_ascii=False),
        })

        # Zincir durumunu güncelle
        if name == "get_master_index":
            state["master_done"] = True
        elif name == "search_decisions":
            state["search_done"] = True
            state["search_empty"] = result.get("toplam_eslesen", 0) == 0

            # Atıf için karar meta bilgilerini sakla (doc_id -> esas_no/mahkeme/karar_tarihi)
            for d in result.get("kararlar", []):
                docs_meta[d["doc_id"]] = d

            # Ollama'nin tool_choice zorlamasi sadece ilk turda guvenilir calisiyor;
            # sonraki turlarda LLM zorlanan tool'u yok sayip eski tool'u tekrar cagirabiliyor
            # (gozlemledigimiz sonsuz donguye yol acan davranis). Bu yuzden get_decision_structure
            # adimini LLM'e sormadan biz burada dogrudan calistirip, sonucu LLM'in kendisi
            # cagirmis gibi gorecegi bir mesaj olarak gecmise ekliyoruz.
            if not state["search_empty"]:
                sonuclar = result.get("kararlar", [])
                if sonuclar:
                    top_doc_id = sonuclar[0]["doc_id"]  # dış kapsamdaki değişkeni güncelle
                    structure_result = dispatch("get_decision_structure", {"doc_id": top_doc_id})

                    if verbose:
                        print(f"  -> (otomatik) get_decision_structure({{'doc_id': '{top_doc_id}'}})")

                    messages.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "function": {
                                "name": "get_decision_structure",
                                "arguments": {"doc_id": top_doc_id},
                            }
                        }],
                    })
                    messages.append({
                        "role": "tool",
                        "content": json.dumps(structure_result, ensure_ascii=False),
                    })

                    if "hata" not in structure_result:
                        state["structure_done"] = True
                        docs_meta[top_doc_id] = structure_result  # karar_no dahil tam meta
        elif name == "get_decision_structure" and "hata" not in result:
            state["structure_done"] = True
        elif name == "get_section" and "hata" not in result:
            state["section_done"] = True
            cited_doc_id = result.get("doc_id")  # cevabın gerçekten hangi karardan üretildiği

    return _with_citation("Maksimum tur sayisina ulasildi.", cited_doc_id, docs_meta)


def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        print("Hukuk Karar Asistani (PageIndex + Ollama llama3.1:8b)")
        print("Cikmak icin 'q' yazin.\n")
        question = input("Sorunuzu yazin: ").strip()

    if question.lower() == "q":
        return

    print(f"\nSoru: {question}")
    print("-" * 60)
    answer = run_query(question, verbose=True)
    print("\n" + "=" * 60)
    print("YANIT:")
    print(answer)


if __name__ == "__main__":
    main()
