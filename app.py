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
    question = data.get("question", "")
    session_id = data.get("session_id")

    if not session_id:
        return jsonify({"error": "Session ID required"}), 400
    if not question.strip():
        return jsonify({"error": "Question required"}), 400

    # Fetch previous chat history
    history = get_history(session_id)

    # Search RAG context
    results = search_index(question, top_k=3)
    if not results:
        return jsonify({"answer": "No documents found. Please upload guidelines first."})

    context = "\n\n".join([chunk for _, chunk in results])

    # Build prompt with history
    history_text = "\n".join(
        [f"User: {h['user']}\nAssistant: {h['assistant']}" for h in history]
    )

    prompt = (
        "You are a clinical assistant. "
        "Answer based on clinical guidelines and prior conversation. "
        "If unsure, say you don't know.\n\n"
        f"Conversation so far:\n{history_text}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    answer = response.text

    # Update history
    history.append({"user": question, "assistant": answer})
    save_history(session_id, history)

    return jsonify({
        "answer": answer,
        "sources": [doc for doc, _ in results],
        "history_length": len(history)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
