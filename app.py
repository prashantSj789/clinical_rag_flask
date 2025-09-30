from io import BytesIO
import json
import os
import uuid
from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
import docx
from flask.cli import load_dotenv
import redis 
from google import genai
from vector_store_to_chroma import add_to_index, search_index
from Bio import Entrez

app = Flask(__name__)

UPLOAD_FOLDER = "data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
load_dotenv()

# Initialize Gemini client (from env or fallback)
key = os.getenv("GEMINI_API_KEY",)
client = genai.Client(api_key=key)
# model name as a string when invoking generate_content
MODEL_NAME = "models/gemini-2.5-flash"

REDIS_URL = "redis://localhost:6379/0"

rdb = redis.Redis.from_url(REDIS_URL, decode_responses=True)

Entrez.email = os.getenv("ENTREZ_EMAIL")  
Entrez.api_key = os.getenv("ENTREZ_API_KEY")

def extract_text(file_path):
    """Extract raw text from PDF, DOCX, or TXT"""
    text = ""
    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    return text

def chunk_text(text, chunk_size=2000, overlap=200):
    """Split text into larger overlapping chunks (fewer records in Chroma)"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def pubmed_search(query, max_results=3):
    """
    Search PubMed for the query, return summary or abstract snippets.
    """
    try:
        # Search for article IDs
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        search_data = Entrez.read(handle)
        handle.close()

        id_list = search_data.get("IdList", [])
        if not id_list:
            return None

        # Fetch article summaries / abstracts
        handle2 = Entrez.efetch(db="pubmed", id=",".join(id_list), retmode="xml")
        fetch_data = Entrez.read(handle2)
        handle2.close()

        results = []
        for article in fetch_data.get("PubmedArticle", []):
            # Extract title, abstract
            try:
                title = article["MedlineCitation"]["Article"]["ArticleTitle"]
            except:
                title = ""
            abstract = ""
            try:
                abstracts = article["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]
                if isinstance(abstracts, list):
                    abstract = " ".join(abstracts)
                else:
                    abstract = str(abstracts)
            except:
                abstract = ""
            snippet = f"Title: {title}\nAbstract: {abstract[:500]}..."
            results.append(snippet)
        return "\n\n".join(results)
    except Exception as e:
        print("PubMed search error:", e)
        return None













@app.route("/createsession", methods=["GET"])
def create_session():
    session_id = str(uuid.uuid4())
    rdb.set(session_id, json.dumps({"history": []}))
    return jsonify({"session_id": session_id})

def get_history(session_id):
    data = rdb.get(session_id)
    if not data:
        return []
    return json.loads(data).get("history", [])


def save_history(session_id, history):
    rdb.set(session_id, json.dumps({"history": history}))

@app.route("/upload", methods=["POST"])
def upload_doc():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    doc_id = file.filename


    file_stream = BytesIO(file.read())
    text = extract_text(file_stream)  
    chunks = chunk_text(text)

    if not chunks:
        return jsonify({"error": "No text extracted from document"}), 400

    add_to_index(chunks, doc_id)
    return jsonify({"message": "Document indexed successfully", "doc_id": doc_id, "chunks": len(chunks)})

@app.route("/query", methods=["POST"])
def query_doc():
    data = request.get_json()
    question = data.get("question", "").strip()
    session_id = data.get("session_id")
    if not session_id or not question:
        return jsonify({"error": "session_id & question needed"}), 400

    history = get_history(session_id)
    # RAG retrieval
    results = search_index(question, top_k=3)
    context_chunks = [chunk for _, chunk in results]
    context_text = "\n\n".join(context_chunks) if context_chunks else ""

    # Build initial prompt, asking LLM whether to use web tool
    history_text = "\n".join(
        [f"User: {h['user']}\nAssistant: {h['assistant']}" for h in history]
    )

    prompt1 = f"""
You are a clinical assistant.  
You have access to 2 sources of information:

1. CONTEXT (from clinical guidelines documents)  
2. A trusted PubMed search tool

If the needed answer is fully covered in the CONTEXT, answer using only CONTEXT.
If CONTEXT is missing or insufficient, respond exactly with the text "NEED_PUBMED".  
Do not hallucinate or guess.  

History:
{history_text}

CONTEXT:
{context_text or '[none]'}

QUESTION:
{question}

If you decide the context is insufficient, reply "NEED_PUBMED". Otherwise answer.
"""

    resp1 = client.models.generate_content(model=MODEL_NAME, contents=prompt1)
    text1 = resp1.text.strip()

    if text1 == "NEED_PUBMED":
        # Use PubMed tool
        pubmed_info = pubmed_search(question, max_results=3)
        if not pubmed_info:
            answer = "I could not find relevant results in PubMed either. I'm sorry, I don't know."
        else:
            prompt2 = f"""
You are a clinical assistant. Use the following PubMed search results to answer the question fully and accurately.  
Be cautious and mention if information is uncertain.

PubMed Results:
{pubmed_info}

QUESTION:
{question}

Answer:
"""
            resp2 = client.models.generate_content(model=MODEL_NAME, contents=prompt2)
            answer = resp2.text.strip()
    else:
        # LLM used context already
        answer = text1

    # Save to history
    history.append({"user": question, "assistant": answer})
    save_history(session_id, history)

    return jsonify({
        "answer": answer,
        "sources": [doc for doc, _ in results],
        "history_length": len(history)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
