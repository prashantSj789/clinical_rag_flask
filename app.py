from io import BytesIO
import os
from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
import docx
from flask.cli import load_dotenv
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

    if not question.strip():
        return jsonify({"error": "Question required"}), 400

    results = search_index(question, top_k=3)
    if not results:
        return jsonify({"answer": "No documents found. Please upload guidelines first."})

    context = "\n\n".join([chunk for _, chunk in results])
    prompt = (
        "You are a clinical assistant. "
        "Given the clinical guidelines below, answer the question. "
        "If unsure, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return jsonify({
        "answer": response.text,
        "sources": [doc for doc, _ in results]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
