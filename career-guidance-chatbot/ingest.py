"""Run once (or after editing data/knowledge): python ingest.py"""
from src.rag import ingest

if __name__ == "__main__":
    print(f"Indexed {ingest()} chunks into ChromaDB.")
