import os
from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
import docx
from google import genai
from vector_store import add_to_index, search_index

app = Flask(__name__)

UPLOAD_FOLDER = "data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Gemini client (will pick up from GEMINI_API_KEY env var)
key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key="AIzaSyDAmZTbNYwq6QA2-luBg0SeuYE7Rb8D6wI")

MODEL_NAME = "gemini-2.5-flash"  # or "gemini-2.5-pro" if available to you

def extract_text(file_path):
    text = ""
    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

@app.route("/upload", methods=["POST"])
def upload_doc():
    file = request.files["file"]
    doc_id = file.filename
    file_path = os.path.join(UPLOAD_FOLDER, doc_id)
    file.save(file_path)

    text = extract_text(file_path)
    chunks = chunk_text(text)
    add_to_index(chunks, doc_id)
    return jsonify({"message": "Document indexed successfully", "doc_id": doc_id})

@app.route("/query", methods=["POST"])
def query_doc():
    data = request.get_json()
    question = data.get("question")

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
