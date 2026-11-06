"""Local SentenceTransformers embeddings (no network; weights come from the local HF cache)."""
from functools import lru_cache

from src.config import settings  # noqa: F401  (must precede sentence_transformers)


class EmbeddingModelUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(settings.EMBEDDING_MODEL)
    except Exception as exc:  # missing cache under HF_HUB_OFFLINE surfaces as several error types
        raise EmbeddingModelUnavailable(
            f"Embedding model {settings.EMBEDDING_MODEL!r} is not in the local cache. "
            "Run once with network access: .venv/bin/python -m scripts.prefetch_model"
        ) from exc


def embed(texts):
    """Unit-normalised vectors as plain lists (Chroma is given embeddings explicitly)."""
    if not texts:
        return []
    return get_model().encode(list(texts), normalize_embeddings=True).tolist()
