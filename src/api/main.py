"""FastAPI app: POST /ask, POST /add-document, WebSocket /chat.

HTTP and WebSocket share one request path (src/pipeline.py). The pipeline is blocking (CrewAI,
embeddings), so every handler runs it with run_in_threadpool and the event loop stays free for
other clients. Every HTTP request (including unknown routes, wrong methods and /docs) and every WebSocket
turn writes exactly one JSONL entry, PII-masked before the write. WebSocket connect/disconnect events are
not separate entries.
"""
import json
import logging
import threading
import time
import uuid

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.concurrency import run_in_threadpool

from src.api.documents import DocumentIndexer, DocumentRejected
from src.config import settings
from src.memory.session_memory import new_session_id
from src.observability.logging_config import RequestLogger
from src.pipeline import ERROR_ANSWER, Pipeline, PipelineResult
from src.schemas.api import AddDocumentRequest, AddDocumentResponse, AskRequest, AskResponse, ErrorResponse

logger = logging.getLogger("uvicorn.error")

HTTP_STATUS = {"answered": 200, "fallback": 200, "clarification": 200, "rejected": 400, "error": 500}


def to_ask_response(r: PipelineResult) -> AskResponse:
    return AskResponse(
        trace_id=r.trace_id, session_id=r.session_id, status=r.status, answer=r.answer, response=r.response,
        sources=r.response.sources if r.response else [], fallback=r.status == "fallback",
        rejection_reason=r.rejection_reason if r.status == "rejected" else None, budget=r.budget, duration_ms=r.duration_ms,
    )


def http_status_for(r: PipelineResult) -> int:
    if r.rejection_reason == "invalid_request":
        return 422
    if r.rejection_reason == "budget_exceeded":
        return 413
    return HTTP_STATUS[r.status]


class Service:
    """Everything the routes need; created lazily so importing the module needs no vector index."""

    def __init__(self, pipeline=None, log_path=None):
        self._pipeline = pipeline
        self._lock = threading.Lock()
        self.logger = RequestLogger(log_path or settings.LOG_PATH)
        self.active_connections = 0
        self._indexer = None

    @property
    def pipeline(self):
        with self._lock:
            if self._pipeline is None:
                self._pipeline = Pipeline()
            return self._pipeline

    @property
    def indexer(self):
        pipe = self.pipeline
        with self._lock:
            if self._indexer is None:
                self._indexer = DocumentIndexer(pipe.collection)
            return self._indexer

    def ask(self, question, session_id, endpoint="/ask", method="POST"):
        trace_id = uuid.uuid4().hex
        try:
            result = self.pipeline.handle(question, session_id, trace_id)
        except Exception as exc:  # last-resort guard: a request must never take the server down
            logger.exception("unhandled error in the request pipeline (trace %s)", trace_id)
            result = PipelineResult(trace_id=trace_id, session_id=session_id or new_session_id(), status="error",
                                    answer=ERROR_ANSWER, rejection_reason=f"unhandled {type(exc).__name__}")
        status = http_status_for(result)
        self.logger.log_result(endpoint, method, result, status)
        return result, status

    def add_document(self, req: AddDocumentRequest):
        trace_id, t0 = uuid.uuid4().hex, time.perf_counter()
        base = dict(trace_id=trace_id, endpoint="/add-document", method="POST", request_text=f"{req.title} | {len(req.text)} chars")
        try:
            doc_id, title, n, n_phone = self.indexer.add(req.title, req.text, req.doc_id)
        except DocumentRejected as exc:
            status = 409 if exc.code == "exists" else 400
            self.logger.log(level="WARNING", status="rejected", http_status=status, rejection_reason=exc.code,
                            duration_ms=round((time.perf_counter() - t0) * 1000, 3), **base)
            return ErrorResponse(trace_id=trace_id, error=exc.code, detail=str(exc)), status
        except Exception as exc:
            logger.exception("unhandled error while indexing a document (trace %s)", trace_id)
            self.logger.log(level="ERROR", status="error", http_status=500, rejection_reason=f"unhandled {type(exc).__name__}",
                            duration_ms=round((time.perf_counter() - t0) * 1000, 3), **base)
            return ErrorResponse(trace_id=trace_id, error="internal_error", detail="document could not be indexed"), 500
        self.pipeline.on_index_changed()
        col = self.pipeline.collection
        self.logger.log(status="indexed", http_status=200, doc_id=doc_id, chunks_indexed=n, phone_numbers_masked=n_phone,
                        duration_ms=round((time.perf_counter() - t0) * 1000, 3), **base)
        return AddDocumentResponse(trace_id=trace_id, doc_id=doc_id, title=title, chunks_indexed=n, collection=col.name,
                                   collection_count=col.count(), phone_numbers_masked=n_phone), 200


def create_app(pipeline=None, log_path=None):
    svc = Service(pipeline, log_path)

    @asynccontextmanager
    async def lifespan(app):
        _ = svc.pipeline  # fail fast at startup, with an actionable message, if the vector index has not been built
        yield

    app = FastAPI(title="Naukri.com Domain Support Agent", lifespan=lifespan)
    app.state.service = svc

    @app.middleware("http")
    async def log_unhandled_requests(request: Request, call_next):
        """Writes the one entry for any HTTP request that no endpoint logged itself (unknown route, wrong method, /docs)."""
        t0 = time.perf_counter()
        request.state.t0 = t0  # read by the validation-error handler so a 422 entry also logs measured timing
        request.state.logged = False  # request.state lives in the ASGI scope, shared with the endpoint and handlers
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            if not request.state.logged:  # no body is read or logged here: unknown routes carry no question
                route = request.scope.get("route")
                svc.logger.log(level="INFO" if status < 400 else "WARNING" if status < 500 else "ERROR",
                               trace_id=uuid.uuid4().hex, endpoint=getattr(route, "path", request.url.path), method=request.method,
                               status="ok" if status < 400 else "rejected" if status < 500 else "error", http_status=status,
                               duration_ms=round((time.perf_counter() - t0) * 1000, 3))

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        trace_id = uuid.uuid4().hex
        t0 = getattr(request.state, "t0", None) or time.perf_counter()
        svc.logger.log(level="WARNING", trace_id=trace_id, endpoint=request.url.path, method=request.method, status="rejected",
                       http_status=422, rejection_reason="invalid_request", duration_ms=round((time.perf_counter() - t0) * 1000, 3),
                       request_text=str(exc.body)[:500])
        request.state.logged = True
        detail = [{"loc": [str(x) for x in e["loc"]], "msg": e["msg"]} for e in exc.errors()]  # never echo the input back
        return JSONResponse(status_code=422, content={"trace_id": trace_id, "error": "invalid_request", "detail": detail})

    @app.post("/ask", response_model=AskResponse)
    async def ask(req: AskRequest, request: Request):
        result, status = await run_in_threadpool(svc.ask, req.question, req.session_id)
        request.state.logged = True
        return JSONResponse(status_code=status, content=to_ask_response(result).model_dump(mode="json"))

    @app.post("/add-document", response_model=AddDocumentResponse, responses={400: {"model": ErrorResponse}, 409: {"model": ErrorResponse}})
    async def add_document(req: AddDocumentRequest, request: Request):
        body, status = await run_in_threadpool(svc.add_document, req)
        request.state.logged = True
        return JSONResponse(status_code=status, content=body.model_dump(mode="json"))

    @app.websocket("/chat")
    async def chat(ws: WebSocket):
        await ws.accept()
        supplied = ws.query_params.get("session_id")
        session_id = supplied or new_session_id()
        svc.active_connections += 1
        await ws.send_json({"type": "session", "session_id": session_id})
        try:
            while True:
                raw = await ws.receive_text()
                question = raw
                try:
                    obj = json.loads(raw)
                    if isinstance(obj, dict) and "question" in obj:
                        question = obj["question"]
                except ValueError:
                    pass
                result, _ = await run_in_threadpool(svc.ask, question, session_id, "/chat", "WEBSOCKET")
                await ws.send_json({"type": "answer", **to_ask_response(result).model_dump(mode="json")})
        except (WebSocketDisconnect, RuntimeError):  # RuntimeError: send on a socket the client already closed
            pass  # a client leaving mid-conversation is normal; nothing else is affected
        finally:
            svc.active_connections -= 1
            if not supplied:  # a server-generated session belongs to this connection only; release it
                svc.pipeline.sessions.drop(session_id)

    return app


app = create_app()
