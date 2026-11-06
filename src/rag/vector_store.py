"""One persistent ChromaDB collection per chunking strategy, populated with upsert()."""
import chromadb
from chromadb.config import Settings

from src.config import settings
from src.rag.chunking import chunk_documents
from src.rag.embeddings import embed


def get_client(path=None):
    return chromadb.PersistentClient(path=str(path or settings.CHROMA_DIR), settings=Settings(anonymized_telemetry=False))


# Chroma 1.5.9 quirk found in testing: when a collection has been read (count/get/query) after its bulk load, a LATER
# small upsert is sometimes stored but never returned by queries (about 1 build in 5 without the settings below).
# Flushing vectors into the HNSW index on every write (batch_size/sync_threshold = 2, the minimum) removed it in 150/150
# trial builds, and `upsert_chunks(verify=True)` additionally proves every written chunk is searchable.
COLLECTION_METADATA = {"hnsw:space": "cosine", "hnsw:batch_size": 2, "hnsw:sync_threshold": 2}
VERIFY_ATTEMPTS = 3


class IndexNotSearchable(RuntimeError):
    pass


def get_collection(client, strategy):
    return client.get_or_create_collection(settings.COLLECTION_NAMES[strategy], metadata=dict(COLLECTION_METADATA))


def _unsearchable(collection, chunks, embeddings):
    missing = []
    for c, e in zip(chunks, embeddings):
        found = collection.query(query_embeddings=[e], n_results=min(3, collection.count()))["ids"][0]
        if c.chunk_id not in found:
            missing.append(c)
    return missing


def upsert_chunks(collection, chunks, verify=False):
    """Upsert chunks (with precomputed embeddings). With verify=True each chunk must then be returned by a query
    using its own embedding; chunks that are not are upserted again, and persistent failure raises IndexNotSearchable."""
    if not chunks:
        return 0
    embeddings = embed([c.text for c in chunks])

    def write(cs, es):
        collection.upsert(ids=[c.chunk_id for c in cs], embeddings=es, documents=[c.text for c in cs], metadatas=[c.metadata() for c in cs])

    write(chunks, embeddings)
    if verify:
        pending = list(zip(chunks, embeddings))
        for _ in range(VERIFY_ATTEMPTS):
            missing = _unsearchable(collection, [c for c, _ in pending], [e for _, e in pending])
            if not missing:
                break
            ids = {c.chunk_id for c in missing}
            pending = [(c, e) for c, e in pending if c.chunk_id in ids]
            write([c for c, _ in pending], [e for _, e in pending])
        else:
            if _unsearchable(collection, [c for c, _ in pending], [e for _, e in pending]):
                raise IndexNotSearchable("written chunks are not returned by the vector index")
    return len(chunks)


def build_collections(client, docs):
    """Index the same documents under both strategies; returns {strategy: (collection, chunks_upserted)}."""
    out = {}
    for strategy in settings.COLLECTION_NAMES:
        col = get_collection(client, strategy)
        out[strategy] = (col, upsert_chunks(col, chunk_documents(docs, strategy)))
    return out
