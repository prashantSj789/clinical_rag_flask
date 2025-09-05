import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "faiss_index/index.faiss"
META_PATH = "faiss_index/metadata.pkl"

# Load embedding model once
embedder = SentenceTransformer(MODEL_NAME)

def embed_texts(texts):
    return embedder.encode(texts, convert_to_numpy=True)

def load_index():
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            metadata = pickle.load(f)
        return index, metadata
    return None, []

def save_index(index, metadata):
    os.makedirs("faiss_index", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)

def add_to_index(text_chunks, doc_id):
    embeddings = embed_texts(text_chunks)

    index, metadata = load_index()
    if index is None:
        index = faiss.IndexFlatL2(embeddings.shape[1])  # New index
        metadata = []

    index.add(embeddings)
    metadata.extend([(doc_id, chunk) for chunk in text_chunks])

    save_index(index, metadata)

def search_index(query, top_k=3):
    index, metadata = load_index()
    if index is None:
        return []

    query_emb = embed_texts([query])
    distances, indices = index.search(query_emb, top_k)

    results = []
    for i in indices[0]:
        if i < len(metadata):
            results.append(metadata[i])
    return results
