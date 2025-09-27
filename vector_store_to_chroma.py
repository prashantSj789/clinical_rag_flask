import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "my_documents"
load_dotenv()


tenant = os.getenv("TENANT").strip()
database = os.getenv("DATABASE").strip()
api_key = os.getenv("CHROMA_API_KEY").strip()

client = chromadb.CloudClient(
    tenant=tenant,
    database=database,
    api_key=api_key,
)

# Create or get collection
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"description": "Document embeddings with SentenceTransformers"},
)

# Load embedding model once
embedder = SentenceTransformer(MODEL_NAME)


# -----------------------------
# Functions
# -----------------------------
def embed_texts(texts):
    return embedder.encode(texts, convert_to_numpy=True).tolist()


def add_to_index(text_chunks, doc_id):

    embeddings = embed_texts(text_chunks)

    collection.add(
        documents=text_chunks,
        embeddings=embeddings,
        ids=[f"{doc_id}_{i}" for i in range(len(text_chunks))],
        metadatas=[{"doc_id": doc_id} for _ in text_chunks],
    )


def search_index(query, top_k=3):
 
    query_emb = embed_texts([query])[0]

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
    )

    matches = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        matches.append((meta["doc_id"], doc))

    return matches


