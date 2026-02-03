import os
import uuid
from dotenv import load_dotenv
from supabase import create_client
from pypdf import PdfReader
from google import genai

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
client = genai.Client(api_key=GOOGLE_API_KEY)

BUCKET = "pdfs"
EMBED_MODEL = "text-embedding-004"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


# ---------- PDF text extraction ----------
def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text.strip()


# ---------- Chunking ----------
def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ---------- Gemini embedding ----------
def get_embedding(text: str):
    res = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text
    )
    return res.embeddings[0].values


# ---------- Upload to Supabase Storage ----------
def upload_pdf_to_supabase(pdf_path: str):
    file_name = os.path.basename(pdf_path)
    unique_name = f"{uuid.uuid4()}_{file_name}"

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    supabase.storage.from_(BUCKET).upload(
        path=unique_name,
        file=pdf_bytes,
        file_options={"content-type": "application/pdf"}
    )

    public_url = supabase.storage.from_(BUCKET).get_public_url(unique_name)
    return unique_name, public_url


# ---------- Ingest One PDF ----------
def ingest_pdf(pdf_path: str):
    print(f"\n📄 Processing: {pdf_path}")

    # Upload
    stored_name, file_url = upload_pdf_to_supabase(pdf_path)
    print("✅ Uploaded to Supabase Storage")
    print("🔗 URL:", file_url)

    # Extract
    text = extract_text_from_pdf(pdf_path)
    if not text:
        print("⚠️ No text extracted, skipping.")
        return

    # Chunk
    chunks = chunk_text(text)
    print(f"✅ Chunks: {len(chunks)}")

    # Insert embeddings
    batch = []
    for i, chunk in enumerate(chunks):
        emb = get_embedding(chunk)

        batch.append({
            "file_name": stored_name,
            "file_url": file_url,
            "chunk_index": i,
            "content": chunk,
            "embedding": emb
        })

        # insert every 15 chunks
        if len(batch) >= 15:
            supabase.table("documents").insert(batch).execute()
            print(f"⬆️ Inserted {i+1}/{len(chunks)} chunks...")
            batch = []

    if batch:
        supabase.table("documents").insert(batch).execute()

    print("🎉 Done:", pdf_path)


# ---------- Ingest All PDFs from folder ----------
def ingest_folder(folder="docs"):
    if not os.path.exists(folder):
        print(f"❌ Folder not found: {folder}")
        return

    pdfs = [f for f in os.listdir(folder) if f.lower().endswith(".pdf")]
    if not pdfs:
        print("⚠️ No PDF files found inside", folder)
        return

    print(f"\n📂 Found {len(pdfs)} PDFs in '{folder}'\n")

    for pdf in pdfs:
        path = os.path.join(folder, pdf)
        try:
            ingest_pdf(path)
        except Exception as e:
            print("❌ Failed:", pdf, "Reason:", str(e))


if __name__ == "__main__":
    ingest_folder("docs")
