"""The only networked step: downloads the embedding model into the local Hugging Face cache.

Usage: .venv/bin/python -m scripts.prefetch_model [--check]
--check only verifies the model loads offline (no network).
"""
import os
import sys

check_only = "--check" in sys.argv
if not check_only:
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

from src.config import settings  # noqa: E402  (setdefault keeps the values set above)
from src.rag.embeddings import embed, get_model  # noqa: E402

if not check_only:
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(settings.EMBEDDING_MODEL)  # downloads on first use
    print(f"downloaded/verified {settings.EMBEDDING_MODEL}")

get_model.cache_clear()
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
dim = len(embed(["offline check"])[0])
print(f"offline load OK ({settings.EMBEDDING_MODEL}, dim={dim})")
