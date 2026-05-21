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
MAX_ROUNDS = 6

SYSTEM = """Sen bir Turk hukuku asistanisin. Elinde mahkeme kararlarindan olusan bir veri tabani var.

ZORUNLU SIRALAMA:
1. get_master_index() -> mevcut dava turlerini ve adedi gor.
2. Eger sorguda gecen kelime (zimmet, bosanma, tazminat vb.) dava_turu_ve_adet listesinde AYNEN varsa: search_decisions(dava_turu="<listedeki deger>") cagir.
   Eger AYNEN yoksa: search_decisions(dava_turu="", keyword="<arama kelimesi>") cagir. dava_turu'nu uydurma.
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

# Bu sistemin llm agent sistemiyle nasıl çalıştığına dair temel akış:
def run_query(question: str, verbose: bool = True) -> str:
    messages = [
        {"role": "system",  "content": SYSTEM}, # LLM'e gönderilen Başlangıç context'i System prompt'u ve kullanıcı sorusu User mesajı olarak başlar.
        {"role": "user",    "content": question},
    ]
    ollama_tools = _to_ollama_tools(TOOLS) # LLM'e TOOL setini verdik

    # Bu döngü agent sistem döngüsü gibi düşünülebilir burada metodları çağırır ve sonuçları LLM'e geri veririz.
    for round_num in range(MAX_ROUNDS):
        #Ollama API request body.
        payload = {
            "model":    MODEL,
            "messages": messages,
            "tools":    ollama_tools,
            "stream":   False,
        }
        # İlk turda get_master_index'i zorunlu kıl - İLK ADIMDA SADECE BU TOOL'U KULLAN
        if round_num == 0:
            payload["tool_choice"] = {
                "type": "function",
                "function": {"name": "get_master_index"}
            }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()

        msg = data.get("message", {})
        tool_calls = msg.get("tool_calls", [])

        if verbose:
            print(f"\n[Tur {round_num + 1}] tool_calls={len(tool_calls)}")

        if not tool_calls:
            return msg.get("content", "")

        # Modelin birden fazla tool çağırmasını engelle: sadece ilk geçerliyi al
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
            # Hiç geçerli tool yoksa modelden düz yanıt iste
            messages.append({"role": "assistant", "content": msg.get("content") or ""})
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

    return "Maksimum tur sayisina ulasildi."


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
