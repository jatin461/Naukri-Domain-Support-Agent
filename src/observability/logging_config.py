"""ELK-style JSON-Lines request log: one flat JSON object per line.

Every string value is passed through the shared `mask_phone` BEFORE the line is written, so a raw
fixed-format phone number never reaches the file (nothing is written and redacted afterwards).
Values must be scalars (flat records); free-form nested dumps are rejected. Answers and secrets are not logged.
"""
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from src.guardrails.input_guardrails import mask_phone

_SCALARS = (str, int, float, bool, type(None))


def utc_timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def render_line(event):
    flat = {}
    for key, value in event.items():
        if not isinstance(value, _SCALARS):
            raise TypeError(f"log field {key!r} must be a scalar, got {type(value).__name__}")
        flat[key] = mask_phone(value) if isinstance(value, str) else value
    return json.dumps(flat, ensure_ascii=False, separators=(",", ":"))


class RequestLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _write(self, line):
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def log(self, **fields):
        event = {"@timestamp": utc_timestamp(), "level": fields.pop("level", "INFO"), "event": "request", **fields}
        self._write(render_line(event))

    def log_result(self, endpoint, method, result, http_status):
        """One entry for a pipeline request (/ask or one WebSocket turn)."""
        groundedness = (result.groundedness or {}).get("action")
        self.log(
            level="INFO" if result.status in ("answered", "fallback", "clarification") else "WARNING" if result.status == "rejected" else "ERROR",
            trace_id=result.trace_id, session_id=result.session_id, endpoint=endpoint, method=method,
            status=result.status, http_status=http_status, duration_ms=result.duration_ms,
            request_text=(result.masked_question or "")[:2000], phone_numbers_masked=result.phone_numbers_masked,
            rejection_reason=result.rejection_reason, injection_matched=",".join(result.injection_matched),
            resolution=result.resolution, groundedness_action=groundedness,
            estimated_tokens=(result.budget or {}).get("estimated_tokens"), token_limit=(result.budget or {}).get("limit"),
            budget_allowed=(result.budget or {}).get("allowed"), simulated_cost_usd=(result.budget or {}).get("simulated_cost_usd"),
            cache_hits=result.cache_events.count("hit"), cache_misses=result.cache_events.count("miss"),
            review_approved=(result.review or {}).get("approved"), review_revised=(result.review or {}).get("revised"),
            tools=",".join(t["tool"] for t in result.tool_invocations),
            sources=",".join(result.response.sources) if result.response else "",
            answer_chars=len(result.answer),
        )

    def read(self):
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
