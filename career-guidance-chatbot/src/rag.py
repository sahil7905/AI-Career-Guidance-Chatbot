"""RAG pipeline: load knowledge base -> chunk -> embed -> store in ChromaDB -> retrieve."""
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

ROOT = Path(__file__).resolve().parent.parent
KB_DIR = ROOT / "data" / "knowledge"
DB_DIR = ROOT / "chroma_db"
COLLECTION = "career_kb"
EMBED_MODEL = "all-MiniLM-L6-v2"


def _collection(reset: bool = False):
    client = chromadb.PersistentClient(path=str(DB_DIR))
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    if reset:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    return client.get_or_create_collection(COLLECTION, embedding_function=embedder)


def chunk_text(text: str, size: int = 120, overlap: int = 25) -> list[str]:
    """Split text into overlapping word windows so context is not lost at boundaries."""
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step) if words[i:i + size]]


def ingest() -> int:
    """(Re)build the vector store from every .md file in data/knowledge."""
    col = _collection(reset=True)
    ids, docs, metas = [], [], []
    for path in sorted(KB_DIR.glob("*.md")):
        for i, chunk in enumerate(chunk_text(path.read_text(encoding="utf-8"))):
            ids.append(f"{path.stem}-{i}")
            docs.append(chunk)
            metas.append({"source": path.stem.replace("_", " ").title()})
    if docs:
        col.add(ids=ids, documents=docs, metadatas=metas)
    return len(docs)


def retrieve(query: str, k: int = 5) -> list[dict]:
    """Return top-k most semantically similar chunks with their source."""
    col = _collection()
    if col.count() == 0:
        ingest()
        col = _collection()
    res = col.query(query_texts=[query], n_results=k)
    return [
        {"text": d, "source": m["source"], "distance": dist}
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]
