"""Builds both ChromaDB collections (idempotent: upsert). Usage: PYTHONPATH=. .venv/bin/python -m scripts.build_index"""
from src.config import settings
from src.rag.kb_loader import load_kb
from src.rag.vector_store import build_collections, get_client


def main():
    built = build_collections(get_client(), load_kb())
    for strategy, (col, n) in built.items():
        print(f"{strategy}: collection={col.name} upserted={n} count={col.count()} dir={settings.CHROMA_DIR}")


if __name__ == "__main__":
    main()
