"""Zero-network enforcement: any non-loopback connect fails the test."""
import src.config.settings  # noqa: F401  (sets env before frameworks import)
import socket
import pytest

_real_connect = socket.socket.connect


def _is_loopback(addr):
    return isinstance(addr, tuple) and addr and addr[0] in ("127.0.0.1", "::1", "localhost")


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def guarded(self, addr):
        if self.family == socket.AF_UNIX or _is_loopback(addr):
            return _real_connect(self, addr)
        raise RuntimeError(f"network blocked in tests: {addr!r}")
    monkeypatch.setattr(socket.socket, "connect", guarded)


@pytest.fixture(scope="session")
def production_collection(tmp_path_factory):
    from src.config import settings
    from src.rag.kb_loader import load_kb
    from src.rag.vector_store import build_collections, get_client

    return build_collections(get_client(tmp_path_factory.mktemp("chroma_session")), load_kb())[settings.RAG_STRATEGY][0]


@pytest.fixture
def core():
    from src.llm.mock_llm import MockLLM

    return MockLLM()


@pytest.fixture
def pipeline(production_collection, core):
    from src.pipeline import Pipeline

    return Pipeline(collection=production_collection, llm_core=core)


@pytest.fixture
def sentence_collection_for_offline(production_collection):
    return production_collection


def pytest_itemcollected(item):
    # acceptance-criterion ids travel into the JUnit XML so scripts/acceptance_report.py can build the matrix from executed tests only
    for marker in item.iter_markers("ac"):
        item.user_properties.append(("ac", marker.args[0]))
