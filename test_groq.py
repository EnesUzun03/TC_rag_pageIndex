import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq

api_key = os.getenv("GROQ_API_KEY")
print("Key yüklendi mi:", api_key is not None)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=api_key,
)

response = llm.invoke("Merhaba, çalışıyor musun?")
print(response.content)