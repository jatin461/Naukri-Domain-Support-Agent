"""Additive document indexing for POST /add-document.

New documents are masked and injection-checked like requests, chunked with the production strategy,
embedded locally and upserted into the production collection. The 12 required documents are never
touched: an existing document ID (required or previously added) is refused, not overwritten.
"""
import hashlib
import re
import logging
import threading
import time
from pathlib import Path

from chromadb.errors import InternalError

from src.config import settings
from src.guardrails.input_guardrails import detect_injection, mask_phone, phone_count
from src.rag.chunking import CHUNKERS
from src.rag.kb_loader import KBDocument
from src.rag.vector_store import upsert_chunks


logger = logging.getLogger("uvicorn.error")
INDEX_ATTEMPTS = 3


class DocumentRejected(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code  # "injection" | "too_long" | "empty" | "exists"


def assign_doc_id(title, text):
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:40] or "doc"
    return f"added_{slug}_{hashlib.sha1((title + chr(0) + text).encode()).hexdigest()[:8]}"


class DocumentIndexer:
    def __init__(self, collection, strategy=None):
        self.collection = collection
        self.strategy = strategy or settings.RAG_STRATEGY
        self._lock = threading.Lock()

    def add(self, title, text, doc_id=None):
        title, text = title.strip(), text.strip()
        n_phone = phone_count(title) + phone_count(text)
        title, text = mask_phone(title), mask_phone(text)
        if not title or not text:
            raise DocumentRejected("empty", "title and text must not be empty")
        if len(text) > settings.MAX_DOCUMENT_CHARS:
            raise DocumentRejected("too_long", f"document text exceeds {settings.MAX_DOCUMENT_CHARS} characters")
        if detect_injection(title).blocked or detect_injection(text).blocked:
            raise DocumentRejected("injection", "document rejected: it looks like a prompt-injection attempt")
        doc_id = doc_id or assign_doc_id(title, text)
        with self._lock:
            if self.collection.get(where={"doc_id": doc_id}, limit=1)["ids"]:
                raise DocumentRejected("exists", f"document {doc_id!r} already exists and is not overwritten")
            doc = KBDocument(doc_id, title, " ".join(text.split()), Path("<api>"))
            chunks = CHUNKERS[self.strategy](doc)
            if not chunks:
                raise DocumentRejected("empty", "document produced no chunks")
            n = self._index(chunks)
        return doc_id, title, n, n_phone

    def _index(self, chunks):
        """Upsert with a searchability check. A transient Chroma storage error (seen once, not reproducible on demand) is retried:
        the upsert is idempotent, so repeating it is safe; the last failure propagates and the API answers 500 with a traceback in the server log."""
        for attempt in range(1, INDEX_ATTEMPTS + 1):
            try:
                return upsert_chunks(self.collection, chunks, verify=True)  # prove the new chunks are searchable
            except InternalError:
                logger.warning("transient Chroma error while indexing (attempt %d of %d)", attempt, INDEX_ATTEMPTS, exc_info=True)
                if attempt == INDEX_ATTEMPTS:
                    raise
                time.sleep(0.2 * attempt)
