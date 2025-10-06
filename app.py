from io import BytesIO
import json
import os
import uuid
from celery import Celery
from flask import Flask, request, jsonify
from PyPDF2 import PdfReader

import docx
from flask.cli import load_dotenv
import redis 
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph,END,START

from vector_store_to_chroma import add_project_docs, add_to_index, search_index, search_project_docs
from Bio import Entrez
from typing_extensions import TypedDict

app = Flask(__name__)

UPLOAD_FOLDER = "data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
load_dotenv()

# Initialize Gemini client (from env or fallback)
key = os.getenv("GEMINI_API_KEY",)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=key)

# model name as a string when invoking generate_content
MODEL_NAME = "models/gemini-2.5-flash"

REDIS_URL = "redis://localhost:6379/0"

rdb = redis.Redis.from_url(REDIS_URL, decode_responses=True)


Entrez.email = os.getenv("ENTREZ_EMAIL")  
Entrez.api_key = os.getenv("ENTREZ_API_KEY")

class RAGstate(TypedDict):
    question: str
    context: str
    pubmed_results: str
    aboutquery: str
    answer: str
    history: str


def run_query_task(question, history_text, session_id):
    state = rag_app.invoke({
        "question": question,
        "context": "",
        "pubmed_results": "",
        "answer": "",
        "history": history_text
    })
    answer = state["answer"]

    # Save in Redis & update history
    history = get_history(session_id)
    history.append({"user": question, "assistant": answer})
    save_history(session_id, history)

    return answer


def retrival_node(state: RAGstate) -> RAGstate:
    results = search_index(state["question"], top_k=3)
    context = "\n\n".join([chunk for _, chunk in results]) if results else ""
    return {"context": context}

def retrival_node2(state: RAGstate) -> RAGstate:
    results = search_project_docs(state["question"], top_k=3)
    context = "\n\n".join([chunk for _, chunk in results]) if results else ""
    return {"context": context}

def decision_node(state: RAGstate) -> RAGstate:
    prompt = f"""
You are an assistant with access to two knowledge sources:

1. **Clinical Guidelines Context** → for medical and clinical queries.  
2. **App & Developer Context** → for questions about this project, its features, or the developer (resume, contact, skills).  

Conversation so far:
{state['history'] or '[none]'}

Retrieved CONTEXT:
{state['context'] or '[none]'}

QUESTION:
{state['question']}

INSTRUCTIONS:
- If the retrieved CONTEXT contains enough information to answer, use ONLY that context to answer.  
- If the question is **clinical** but the CONTEXT is insufficient, reply exactly with: "NEED_PUBMED".  
- If the question is about the **app or developer** but the CONTEXT is insufficient, reply exactly with: "AboutQuery".  
- Do NOT use your own general knowledge. Do NOT hallucinate.
"""
    resp = llm.invoke(prompt)
    text = resp.strip() if isinstance(resp, str) else getattr(resp, "content", "").strip()

    if text == "NEED_PUBMED":
        return {"pubmed_results": "NEED"}
    elif text == "AboutQuery":
        return {"aboutquery": "NEED"}
    else:
        return {"answer": text}


    

def pubmed_node(state: RAGstate) -> RAGstate:
    if state.get("pubmed_results") != "NEED":
        return {}
    results = pubmed_search(state["question"], max_results=3)
    return {"pubmed_results": results or ""}   

def doc_node(state: RAGstate) -> RAGstate:
    if state.get("aboutquery") != "NEED":
        return {}
    results = retrival_node2(state)
    return {"context": results.get("context","")}




def answer_node(state: RAGstate) -> RAGstate:
    if state.get("answer"):
        return {}

    prompt = f"""
You are a clinical assistant. 
IMPORTANT RULES:
- ONLY use the provided context to answer.
- If the context does not contain enough information, respond with exactly: "I don't know based on the available context."
- If you do not have enough information to answer from the context you may reffer to the PubMed results  given below if availale.
- Do NOT use your own general knowledge.
- Do NOT invent answers.

Conversation so far:
{state['history'] or '[none]'}

Context:
{state['context'] or '[none]'}

PubMed Results:
{state['pubmed_results'] or '[none]'}

Question:
{state['question']}
"""
    resp = llm.invoke(prompt) 
    text = resp.content.strip()
    return {"answer": text}         


graph = StateGraph(RAGstate)

graph.add_node("retrieval", retrival_node)
graph.add_node("decider", decision_node)
graph.add_node("pubmed", pubmed_node)
graph.add_node("answer", answer_node)
graph.add_node("docretrieval", doc_node)
# flow
graph.add_edge(START, "retrieval")
graph.add_edge("retrieval", "decider")

def decider_branch(state):
    if state.get("pubmed_results") == "NEED":
        return "pubmed"
    elif state.get("aboutquery") == "NEED":
        return "docretrieval"
    else:
        return "answer"

# conditional branch
graph.add_conditional_edges(
    "decider",
    decider_branch,
    {"pubmed": "pubmed", "docretrieval": "docretrieval", "answer": "answer"}
)

graph.add_edge("pubmed", "answer")
graph.add_edge("docretrieval", "answer")
graph.add_edge("answer", END)

rag_app = graph.compile()

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


def get_history(session_id):
    data = rdb.get(session_id)
    if not data:
        return []
    return json.loads(data).get("history", [])

def save_history(session_id, history):
    rdb.set(session_id, json.dumps({"history": history}))








 



@app.route("/createsession", methods=["GET"])
def create_session():
    session_id = str(uuid.uuid4())
    rdb.set(session_id, json.dumps({"history": []}))
    return jsonify({"session_id": session_id})




@app.route("/upload", methods=["POST"])
def upload_doc():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    doc_id = file.filename

    # Save temporarily
    file_path = os.path.join(UPLOAD_FOLDER, doc_id)
    file.save(file_path)

    try:
        text = extract_text(file_path)
    finally:
        # Always remove the temp file
        if os.path.exists(file_path):
            os.remove(file_path)

    chunks = chunk_text(text)
    if not chunks:
        return jsonify({"error": "No text extracted from document"}), 400

    add_to_index(chunks, doc_id)
    return jsonify({
        "message": "Document indexed successfully",
        "doc_id": doc_id,
        "chunks": len(chunks)
    })


@app.route("/upload_docs", methods=["POST"])
def upload_app_doc():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    doc_id = file.filename

    # Save temporarily
    file_path = os.path.join(UPLOAD_FOLDER, doc_id)
    file.save(file_path)

    try:
        text = extract_text(file_path)
    finally:
        # Always remove the temp file
        if os.path.exists(file_path):
            os.remove(file_path)

    chunks = chunk_text(text)
    if not chunks:
        return jsonify({"error": "No text extracted from document"}), 400

    add_project_docs(chunks, doc_id)
    return jsonify({
        "message": "App document indexed successfully",
        "doc_id": doc_id,
        "chunks": len(chunks)
    })

 


@app.route("/query", methods=["POST"])
def query_doc():
    data = request.get_json()
    question = data.get("question", "").strip()
    session_id = data.get("session_id")

    if not session_id or not question:
        return jsonify({"error": "session_id & question needed"}), 400

    history = get_history(session_id)
    history_text = "\n".join([f"User: {h['user']}\nAssistant: {h['assistant']}" for h in history])

    answer = run_query_task(question, history_text, session_id)

    history = get_history(session_id)
    return jsonify({"answer": answer, "history_length": len(history)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)




