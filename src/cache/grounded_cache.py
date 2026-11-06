"""In-memory cache for the grounded-generation step only (retrieval + context-only generation).

Key: (index_version, normalize_query(query)). Normalisation is purely lexical: strip leading and trailing
whitespace and punctuation, lower-case, collapse internal whitespace, keep internal punctuation (so
`APP-0007` keeps its hyphen). It matches variants that differ only in case, surrounding whitespace or surrounding
punctuation; it does NOT match paraphrases and makes no semantic-equivalence claim. The index version changes
whenever a document is added, so cached results never outlive the index they came from.
The cache sits inside the RAG tool, so input guardrails, the budget check, session handling, schema
validation, the groundedness guard and the AutoGen review all still run on every request.
"""
import re
import threading

_EDGES = re.compile(r"^[\W_]+|[\W_]+$")
_SPACES = re.compile(r"\s+")


def normalize_query(text):
    return _SPACES.sub(" ", _EDGES.sub("", text.strip())).lower()


class GroundedCache:
    def __init__(self):
        self._data = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get_or_compute(self, query, version, compute):
        """Returns (value, "hit" | "miss"). `compute()` runs outside the lock; concurrent misses on one key may
        both compute, but only one value is kept and every caller gets a consistent result."""
        key = (version, normalize_query(query))
        with self._lock:
            if key in self._data:
                self.hits += 1
                return self._data[key], "hit"
        value = compute()
        with self._lock:
            self.misses += 1
            value = self._data.setdefault(key, value)
        return value, "miss"

    def clear(self):
        with self._lock:
            self._data.clear()

    def stats(self):
        with self._lock:
            return {"hits": self.hits, "misses": self.misses, "size": len(self._data)}
