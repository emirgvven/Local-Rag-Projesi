import os
import sqlite3
import math
import json
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Çalışan motorumuz: Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

# --- 1. API VE GÜVENLİK AYARLARI ---
app = FastAPI(title="Kurumsal Local RAG API (SQLite + Ollama)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. OLLAMA VE SQLITE KURULUMU ---
print("Yapay Zeka Motoru Bağlanıyor (Ollama)...")
embed_model = OllamaEmbeddings(model="nomic-embed-text")
chat_model = Ollama(model="llama3")

# Müfredatta istenen yerel veritabanı (SQLite)
DB_PATH = "rag_database.sqlite"

def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS documents 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT, embedding TEXT)''')
    conn.commit()
    return conn, c

conn, cursor = init_db()

# Metni sayısal vektörlere çevirme (Ollama üzerinden)
def get_embedding(text):
    return embed_model.embed_query(text)

# Müfredat gereksinimi: Saf Python ile Kosinüs Benzerliği Hesaplama
def cosine_similarity(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = math.sqrt(sum(a * a for a in v1))
    norm_v2 = math.sqrt(sum(b * b for b in v2))
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    return dot_product / (norm_v1 * norm_v2)

# --- 3. DOKÜMANLARI İÇERİ AKTARMA ---
docs_klasoru = './docs'
if not os.path.exists(docs_klasoru):
    os.makedirs(docs_klasoru)
    with open(os.path.join(docs_klasoru, "ornek_veri.md"), "w", encoding="utf-8") as f:
        f.write("DBS (Derin Beyin Stimülasyonu) ve VNS (Vagus Sinir Stimülasyonu) nöromodülasyon teknolojileridir. DBS beyni, VNS vagus sinirini hedefler.")

cursor.execute("SELECT COUNT(*) FROM documents")
if cursor.fetchone()[0] == 0:
    print("Dokümanlar okunuyor ve SQLite veritabanına indeksleniyor...")
    loader = DirectoryLoader(docs_klasoru, glob="**/*.md", loader_cls=TextLoader)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=25)
    splits = text_splitter.split_documents(docs)
    
    for chunk in splits:
        emb = get_embedding(chunk.page_content)
        cursor.execute("INSERT INTO documents (content, embedding) VALUES (?, ?)", 
                       (chunk.page_content, json.dumps(emb)))
    conn.commit()
    print("Veritabanı başarıyla oluşturuldu ve vektörler kaydedildi!")

def get_top_chunks(query, k=3):
    query_emb = get_embedding(query)
    cursor.execute("SELECT content, embedding FROM documents")
    rows = cursor.fetchall()
    
    results = []
    for row in rows:
        content = row[0]
        doc_emb = json.loads(row[1])
        sim = cosine_similarity(query_emb, doc_emb)
        results.append((sim, content))
    
    results.sort(key=lambda x: x[0], reverse=True)
    return [res[1] for res in results[:k]]

# --- 4. SOHBET HAFIZASI VE İSTEK (ENDPOINT) ---
class SoruIstegi(BaseModel):
    soru: str

sohbet_gecmisi = []

@app.post("/chat")
async def chat_endpoint(istek: SoruIstegi, request: Request):
    global sohbet_gecmisi
    try:
        gecmis_metni = ""
        for mesaj in sohbet_gecmisi:
            gecmis_metni += f"Kullanıcı: {mesaj['soru']}\nAsistan: {mesaj['cevap']}\n"

        arama_metni = istek.soru
        if len(sohbet_gecmisi) > 0 and len(istek.soru.split()) < 4:
            arama_metni = f"Önceki konu olan '{sohbet_gecmisi[-1]['soru']}' hakkında: {istek.soru}"

        ilgili_metinler = get_top_chunks(arama_metni)
        baglam = "\n".join(ilgili_metinler)
        if not baglam.strip():
            baglam = "Kayıtlı doküman bulunamadı."

        system_prompt = f"""Sen net, kısa ve doğrudan cevaplar veren bir asistansın.

KURAL 1: Sadece aşağıdaki bağlamı kullan. Bilgi yoksa "Dökümanda bu bilgi yok." de.
KURAL 2: SADECE TÜRKÇE YANIT VER. İngilizce kelime, İngilizce cümle veya parantez içi çeviri KESİNLİKLE KULLANMA. Yabancı terimlerin sadece Türkçe karşılıklarını yaz.
KURAL 3: Sadece doğrudan bilgiyi ver.
KURAL 4: Konunun tanımını tekrar etme. Direkt listelemeye başla.
KURAL 5: Kullanıcı ek bilgi istiyorsa ve bağlamda yoksa sadece "Dökümanda bu konuyla ilgili başka bir bilgi bulunmuyor." de.

Sohbet Geçmişi:
{gecmis_metni}

Bağlam:
{baglam}

Soru: {istek.soru}
Cevap (DİREKT VE GİRİŞSİZ):"""

        # Llama 3 motoruna promptu gönderiyoruz
        alinan_cevap = chat_model.invoke(system_prompt)

        if await request.is_disconnected():
            return {"cevap": "İşlem iptal edildi"}

        sohbet_gecmisi.append({"soru": istek.soru, "cevap": alinan_cevap})
        if len(sohbet_gecmisi) > 3:
            sohbet_gecmisi.pop(0)

        return {"cevap": alinan_cevap}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 5. SUNUCU DURUMU (NABIZ) ---
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Sunucu Aktif"}