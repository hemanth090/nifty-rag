"""Nifty 50 annual-report Q&A server.

A small, single-purpose HTTP service over a local PageIndex store: it serves
the chat UI, lists the indexed documents, answers questions with page-level
citations, and streams answers over SSE. Blocking PageIndex SDK calls run in
Starlette's threadpool so one slow answer does not stall other requests, and
in-flight agent runs are bounded by a semaphore.

Run from the repository root:

    .venv/bin/python server.py              # HOST/PORT/WORKERS from env
    ./run_server.sh                         # same, with defaults

http://127.0.0.1:8000

Configuration comes from ``.env`` in this directory, then the process
environment:

    OPENCODE_GO_API_KEY    required   OpenCode Go API key
    OPENCODE_GO_BASE_URL   optional   default: https://opencode.ai/inference/go/openai/v1
    PAGEINDEX_MODEL        optional   default: deepseek-v4.1-flash
    PAGEINDEX_STORE        optional   default: ./.pageindex

  Request limits:
    MAX_CONCURRENT         optional   in-flight agent runs per worker (8)
    QUEUE_TIMEOUT          optional   seconds to wait for a slot (45)
    MAX_BODY_BYTES         optional   request body limit (65536)
    MAX_QUESTION_CHARS     optional   question length limit (1000)
    MAX_DOC_IDS            optional   document-ID count limit (64)

  Agent and retrieval budgets:
    MAX_TURNS              optional   agent turns per question (10)
    MAX_OUTPUT_TOKENS      optional   per-turn output cap; 0 = provider default (2048)
    TOOL_RESPONSE_CHARS    optional   tool reply budget in characters (100000)
    TOOL_MAX_PAGES         optional   pages one get_page_content call may request (10000)
    TOOL_SUGGESTIONS       optional   similar-name suggestions on a miss (3)
    CATALOG_CHARS          optional   document-list prompt budget in characters (6000)
    DOCUMENT_CATALOG       optional   "0"/"false"/"no" disables the library listing

  Caching and operations:
    ANSWER_CACHE_TTL       optional   answer-cache seconds (3600); 0 disables
    ANSWER_CACHE_MAX       optional   cached answers per worker (500)
    API_TOKEN              optional   when set, required as "Authorization: Bearer <token>"
    HOST / PORT            optional   127.0.0.1 / 8000
    WORKERS                optional   uvicorn workers (1)
    LOG_LEVEL              optional   INFO
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import time
import traceback
import uuid
from collections import OrderedDict, deque
from functools import partial
from pathlib import Path
from typing import AsyncIterator

import uvicorn
from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from pageindex import PageIndexClient, agent_tools
from pageindex.errors import PageIndexAPIError

APP_VERSION = "1.0.0"
STARTED_AT = time.monotonic()

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


# ── Settings ─────────────────────────────────────────────────────────────

def _env_int(name: str, default: int, minimum: int | None = None) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        value = default
    else:
        try:
            value = int(raw)
        except ValueError:
            raise SystemExit(f"{name} must be an integer, got {raw!r}.")
    if minimum is not None and value < minimum:
        raise SystemExit(f"{name} must be >= {minimum}, got {value}.")
    return value


def _env_float(name: str, default: float, minimum: float | None = None) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        value = default
    else:
        try:
            value = float(raw)
        except ValueError:
            raise SystemExit(f"{name} must be a number, got {raw!r}.")
    if minimum is not None and value < minimum:
        raise SystemExit(f"{name} must be >= {minimum}, got {value}.")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return raw.strip() if raw and raw.strip() else default


MODEL = _env_str("PAGEINDEX_MODEL", "deepseek-v4.1-flash")
BASE_URL = _env_str("OPENCODE_GO_BASE_URL",
                    "https://opencode.ai/inference/go/openai/v1").rstrip("/")
STORE = Path(_env_str("PAGEINDEX_STORE", str(ROOT / ".pageindex"))).expanduser()
PAGE = ROOT / "index.html"

MAX_CONCURRENT = _env_int("MAX_CONCURRENT", 8, minimum=1)
QUEUE_TIMEOUT = _env_float("QUEUE_TIMEOUT", 45.0, minimum=0.1)
CACHE_TTL = _env_int("ANSWER_CACHE_TTL", 3600, minimum=0)
CACHE_MAX = _env_int("ANSWER_CACHE_MAX", 500, minimum=0)
USE_CATALOG = _env_bool("DOCUMENT_CATALOG", True)
API_TOKEN = _env_str("API_TOKEN", "")
MAX_BODY_BYTES = _env_int("MAX_BODY_BYTES", 65_536, minimum=1)
MAX_QUESTION_CHARS = _env_int("MAX_QUESTION_CHARS", 1_000, minimum=1)
MAX_DOC_IDS = _env_int("MAX_DOC_IDS", 64, minimum=1)

# Agent and retrieval budgets. These are the knobs that decide how much text
# one question may pull into the model.
MAX_TURNS = _env_int("MAX_TURNS", 10, minimum=1)
MAX_OUTPUT_TOKENS = _env_int("MAX_OUTPUT_TOKENS", 2_048, minimum=0)
TOOL_RESPONSE_CHARS = _env_int("TOOL_RESPONSE_CHARS", 100_000, minimum=1_000)
TOOL_MAX_PAGES = _env_int("TOOL_MAX_PAGES", 10_000, minimum=1)
TOOL_SUGGESTIONS = _env_int("TOOL_SUGGESTIONS", 3, minimum=0)
CATALOG_CHARS = _env_int("CATALOG_CHARS", 6_000, minimum=0)

HOST = _env_str("HOST", "127.0.0.1")
PORT = _env_int("PORT", 8000, minimum=1)
WORKERS = _env_int("WORKERS", 1, minimum=1)
LOG_LEVEL = _env_str("LOG_LEVEL", "INFO").upper()

REASONING_LEVELS = {"none", "low", "medium", "high"}

# Apply the configured tool budgets to the SDK's in-process tool layer. The
# tool code reads these module constants at call time, so setting them here
# changes enforced limits rather than only documentation.
agent_tools.TOOL_RESPONSE_CHAR_LIMIT = TOOL_RESPONSE_CHARS
agent_tools._CHAR_BUDGET = int(TOOL_RESPONSE_CHARS * 0.95)
agent_tools._MAX_REQUESTED_PAGES = TOOL_MAX_PAGES
agent_tools._SIMILAR_NAMES_LIMIT = TOOL_SUGGESTIONS


# ── Logging ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
log = logging.getLogger("nifty_rag.server")


# ── Provider + client ────────────────────────────────────────────────────

def sdk_version() -> str:
    try:
        from importlib.metadata import version
        return version("pageindex")
    except Exception:  # noqa: BLE001 - metadata is optional at runtime
        return "unknown"


def provider_headers() -> dict[str, str]:
    """Headers OpenCode Go asks coding agents to send (session + user agent)."""
    return {
        "User-Agent": f"nifty-rag/{APP_VERSION} (coding-agent)",
        "HTTP-Referer": "https://opencode.ai",
        "x-opencode-session": f"nifty-rag-{uuid.uuid4().hex}",
    }


def build_client() -> PageIndexClient:
    key = _env_str("OPENCODE_GO_API_KEY", "")
    if not key:
        raise SystemExit(
            f"OPENCODE_GO_API_KEY is not set. Add it to {ROOT / '.env'} "
            "or export it in the environment."
        )
    return PageIndexClient(
        index={"model": MODEL, "storage_path": str(STORE)},
        chat={"model": MODEL, "backend": {"api_key": key, "base_url": BASE_URL}},
    )


def load_home() -> str:
    try:
        return PAGE.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"Cannot read the UI file {PAGE}: {exc}") from exc


CLIENT = build_client()
HEADERS = provider_headers()
HOME = load_home()

# ── Runtime state ────────────────────────────────────────────────────────

slots = asyncio.Semaphore(MAX_CONCURRENT)
cache: "OrderedDict[str, tuple[float, dict]]" = OrderedDict()
inflight: dict[str, "asyncio.Future[dict]"] = {}
catalog_cache: dict[str, str] = {"signature": "", "text": ""}

RETRIEVAL_FAILED = "Retrieval failed. Please retry in a moment."

metrics = {
    "requests": 0, "errors": 0, "busy": 0, "cache_hits": 0, "cache_misses": 0,
    "active": 0, "durations_ms": deque(maxlen=512),
}


def store_signature() -> str:
    """Changes whenever the local manifest is rewritten (re-index)."""
    try:
        stat = (STORE / "manifest.json").stat()
    except OSError:
        return "0"
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def document_catalog() -> str:
    """A compact library listing, injected once per question to save browse calls."""
    if not USE_CATALOG:
        return ""
    signature = store_signature()
    if catalog_cache["signature"] == signature:
        return catalog_cache["text"]
    try:
        listed = CLIENT.list_documents(limit=10000)["documents"]
    except Exception as exc:  # noqa: BLE001 - a listing failure must not block answers
        log.warning("document catalog unavailable: %s", exc)
        return ""
    entries = [f"{doc['name']} ({doc.get('pageNum') or '?'} pages)" for doc in listed]
    text = ""
    if entries:
        text = ("The library contains these documents: "
                + "; ".join(entries)
                + ". When the question refers to a company or file, read that "
                  "document directly instead of browsing the library.")
        if len(text) > CATALOG_CHARS:
            text = ""
    catalog_cache["signature"], catalog_cache["text"] = signature, text
    return text


def cache_key(question: str, doc_ids: list[str] | None, effort: str | None) -> str:
    payload = json.dumps({
        "question": " ".join(question.lower().split()),
        "doc_ids": sorted(doc_ids or []),
        "effort": effort or "",
        "model": MODEL,
        "catalog": USE_CATALOG,
        "store": store_signature(),
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def cache_get(key: str) -> dict | None:
    if CACHE_TTL <= 0 or CACHE_MAX <= 0:
        return None
    entry = cache.get(key)
    if entry is None:
        return None
    stored_at, value = entry
    if time.monotonic() - stored_at > CACHE_TTL:
        cache.pop(key, None)
        return None
    cache.move_to_end(key)  # LRU: a hit makes the entry the newest
    metrics["cache_hits"] += 1
    return value


def cache_put(key: str, value: dict) -> None:
    if CACHE_TTL <= 0 or CACHE_MAX <= 0:
        return
    cache[key] = (time.monotonic(), value)
    cache.move_to_end(key)
    while cache and (len(cache) > CACHE_MAX or
                     time.monotonic() - next(iter(cache.values()))[0] > CACHE_TTL):
        cache.popitem(last=False)


def note_duration(started: float) -> None:
    metrics["durations_ms"].append(round((time.monotonic() - started) * 1000))


# ── Single-flight ─────────────────────────────────────────────────────────
# Identical questions arriving together share one agent run instead of paying
# for the same retrieval twice.

def inflight_begin(key: str) -> "asyncio.Future[dict]":
    future: "asyncio.Future[dict]" = asyncio.get_running_loop().create_future()
    inflight[key] = future
    return future


def inflight_finish(key: str, future: "asyncio.Future[dict]",
                    result: dict | None = None, error: str | None = None) -> None:
    if not future.done():
        future.set_result(result if error is None else {"__error__": error})
    if inflight.get(key) is future:
        inflight.pop(key, None)


async def inflight_wait(future: "asyncio.Future[dict]") -> dict:
    return await asyncio.shield(future)


def percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    return values[min(len(values) - 1, int(len(values) * fraction))]


def uptime_seconds() -> int:
    return int(time.monotonic() - STARTED_AT)


# ── Retrieval ────────────────────────────────────────────────────────────

def read_citations(answer: str) -> list[dict]:
    try:
        found = CLIENT.get_citations(answer)
    except Exception as exc:  # noqa: BLE001 - citations are optional metadata
        log.warning("citation parsing failed: %s", exc)
        return []
    return [
        {"document": cite.get("document"), "page": cite.get("page"),
         "doc_id": cite.get("doc_id")}
        for cite in found
    ]


def chat_kwargs(doc_ids: list[str] | None, effort: str | None) -> dict:
    kwargs = {"doc_id": doc_ids or None, "extra_headers": HEADERS,
              "citations": True, "reasoning_effort": effort,
              "max_turns": MAX_TURNS,
              "instructions": document_catalog() or None}
    if MAX_OUTPUT_TOKENS > 0:
        # The chat lane takes output caps under the wire field name; the SDK's
        # extra_body channel forwards it unchanged on the OpenAI protocol.
        kwargs["extra_body"] = {"max_tokens": MAX_OUTPUT_TOKENS}
    return kwargs


def answer_question(question: str, doc_ids: list[str] | None,
                    effort: str | None) -> dict:
    """One blocking tree search; returns {answer, citations}."""
    answer = CLIENT.chat(question, **chat_kwargs(doc_ids, effort))
    return {"answer": answer, "citations": read_citations(answer)}


class RequestProblem(Exception):
    """A client-side request error carrying the HTTP status to return."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def parse_question(body: dict) -> tuple[str, list[str] | None, str | None]:
    question = body.get("question")
    if not isinstance(question, str) or not question.strip():
        raise RequestProblem(400, "The question is empty.")
    question = question.strip()
    if len(question) > MAX_QUESTION_CHARS:
        raise RequestProblem(
            400, f"The question is longer than {MAX_QUESTION_CHARS} characters.")

    raw_ids = body.get("doc_ids") or []
    if not isinstance(raw_ids, list):
        raise RequestProblem(400, "doc_ids must be a list of document IDs, or omitted.")
    if len(raw_ids) > MAX_DOC_IDS:
        raise RequestProblem(400, f"At most {MAX_DOC_IDS} document IDs are allowed.")
    if not all(isinstance(doc_id, str) for doc_id in raw_ids):
        raise RequestProblem(400, "doc_ids must contain strings only.")
    doc_ids = [doc_id.strip() for doc_id in raw_ids if doc_id.strip()]

    effort = body.get("reasoning_effort")
    if effort is not None and effort not in REASONING_LEVELS:
        raise RequestProblem(
            400, f"reasoning_effort must be one of {sorted(REASONING_LEVELS)}, "
                 "or omitted.")
    return question, doc_ids or None, effort


async def read_json(request: Request) -> dict:
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type not in ("application/json", ""):
        raise RequestProblem(415, "Content-Type must be application/json.")
    declared = request.headers.get("content-length")
    if declared:
        try:
            declared_length = int(declared)
        except ValueError as exc:
            raise RequestProblem(400, "Invalid Content-Length header.") from exc
        if declared_length > MAX_BODY_BYTES:
            raise RequestProblem(413, f"Request body exceeds {MAX_BODY_BYTES} bytes.")
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > MAX_BODY_BYTES:
            raise RequestProblem(413, f"Request body exceeds {MAX_BODY_BYTES} bytes.")
    try:
        body = json.loads(bytes(raw))
    except json.JSONDecodeError as exc:
        raise RequestProblem(400, "Request body must be valid JSON.") from exc
    if not isinstance(body, dict):
        raise RequestProblem(400, "Request body must be a JSON object.")
    return body


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def acquire_slot() -> bool:
    """A free agent slot, or False when the queue is too long."""
    try:
        await asyncio.wait_for(slots.acquire(), timeout=QUEUE_TIMEOUT)
        metrics["active"] += 1
        return True
    except TimeoutError:
        metrics["busy"] += 1
        return False


def release_slot(started: float) -> None:
    metrics["active"] = max(0, metrics["active"] - 1)
    note_duration(started)
    slots.release()


BUSY = {"error": "The server is handling too many questions right now. "
                 "Please retry in a moment."}


# ── Middleware ───────────────────────────────────────────────────────────

PROTECTED_PREFIXES = ("/api/", "/metrics")
REQUEST_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")


def harden(response: Response, request_id: str) -> Response:
    """Attach the request ID and the standard security headers to a response."""
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers.setdefault("Cache-Control", "no-store")
    return response


class OperationsMiddleware(BaseHTTPMiddleware):
    """Request IDs, optional bearer auth, security headers, and access logging."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        candidate = request.headers.get("x-request-id", "")
        request_id = (candidate if REQUEST_ID_RE.fullmatch(candidate)
                      else uuid.uuid4().hex[:12])
        started = time.monotonic()
        path = request.url.path

        if API_TOKEN and path.startswith(PROTECTED_PREFIXES):
            supplied = request.headers.get("authorization", "").encode(
                "utf-8", "replace")
            expected = f"Bearer {API_TOKEN}".encode("utf-8", "replace")
            if not hmac.compare_digest(supplied, expected):
                log.warning("auth rejected method=%s path=%s request_id=%s",
                            request.method, path, request_id)
                return harden(JSONResponse({"error": "Unauthorized."},
                                           status_code=401), request_id)

        try:
            response = await call_next(request)
        except Exception:  # noqa: BLE001 - last resort before Starlette's handler
            log.exception("unhandled error method=%s path=%s request_id=%s",
                          request.method, path, request_id)
            metrics["errors"] += 1
            response = JSONResponse(
                {"error": "Internal server error."}, status_code=500)

        elapsed_ms = round((time.monotonic() - started) * 1000, 1)
        harden(response, request_id)
        log.info("request method=%s path=%s status=%s ms=%s request_id=%s",
                 request.method, path, response.status_code, elapsed_ms, request_id)
        return response


# ── Routes ───────────────────────────────────────────────────────────────

async def home(_request: Request):
    return HTMLResponse(HOME)


async def health(_request: Request):
    return JSONResponse({
        "ok": True,
        "version": APP_VERSION,
        "sdk_version": sdk_version(),
        "model": MODEL,
        "store": str(STORE),
        "uptime_s": uptime_seconds(),
        "cache_size": len(cache),
        "active": metrics["active"],
        "budgets": {
            "max_turns": MAX_TURNS,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "tool_response_chars": TOOL_RESPONSE_CHARS,
            "tool_max_pages": TOOL_MAX_PAGES,
            "catalog_chars": CATALOG_CHARS,
        },
    })


async def ready(_request: Request):
    ready_now = (STORE / "manifest.json").is_file()
    return JSONResponse(
        {"ready": ready_now,
         "store": str(STORE),
         "documents": len(list((STORE / "docs").glob("*/doc.json")))
         if (STORE / "docs").is_dir() else 0},
        status_code=200 if ready_now else 503)


async def metrics_endpoint(_request: Request):
    durations = sorted(metrics["durations_ms"])
    return JSONResponse({
        "requests": metrics["requests"],
        "errors": metrics["errors"],
        "busy": metrics["busy"],
        "cache_hits": metrics["cache_hits"],
        "cache_misses": metrics["cache_misses"],
        "cache_size": len(cache),
        "cache_max": CACHE_MAX,
        "active": metrics["active"],
        "max_concurrent": MAX_CONCURRENT,
        "p50_ms": percentile(durations, 0.50),
        "p95_ms": percentile(durations, 0.95),
        "samples": len(durations),
        "model": MODEL,
        "uptime_s": uptime_seconds(),
        "budgets": {
            "max_concurrent": MAX_CONCURRENT,
            "queue_timeout_s": QUEUE_TIMEOUT,
            "max_turns": MAX_TURNS,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "tool_response_chars": TOOL_RESPONSE_CHARS,
            "tool_max_pages": TOOL_MAX_PAGES,
            "catalog_chars": CATALOG_CHARS,
            "max_body_bytes": MAX_BODY_BYTES,
            "max_question_chars": MAX_QUESTION_CHARS,
            "max_doc_ids": MAX_DOC_IDS,
            "cache_ttl_s": CACHE_TTL,
            "cache_max": CACHE_MAX,
        },
    })


async def documents(_request: Request):
    listed = await run_in_threadpool(partial(CLIENT.list_documents, limit=10000))
    docs = [{"id": doc["id"], "name": doc["name"], "pages": doc.get("pageNum")}
            for doc in listed["documents"]]
    return JSONResponse({"documents": docs})


def replay_result(result: dict) -> StreamingResponse:
    """SSE frames for an answer that is already on hand (cache or single-flight)."""
    async def replay() -> AsyncIterator[str]:
        yield sse({"delta": result.get("answer", "")})
        yield sse({**result, "done": True})
    return StreamingResponse(replay(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


async def ask(request: Request):
    started = time.monotonic()
    metrics["requests"] += 1
    try:
        body = await read_json(request)
        question, doc_ids, effort = parse_question(body)
    except RequestProblem as exc:
        metrics["errors"] += 1
        return JSONResponse({"error": str(exc)}, status_code=exc.status_code)

    key = cache_key(question, doc_ids, effort)
    cached = cache_get(key)
    if cached is not None:
        note_duration(started)
        return JSONResponse({**cached, "cached": True})
    metrics["cache_misses"] += 1

    pending = inflight.get(key)
    if pending is not None:
        outcome = await inflight_wait(pending)
        if "__error__" in outcome:
            metrics["errors"] += 1
            return JSONResponse({"error": RETRIEVAL_FAILED}, status_code=502)
        note_duration(started)
        return JSONResponse(outcome)

    future = inflight_begin(key)
    try:
        if not await acquire_slot():
            inflight_finish(key, future, error="busy")
            return JSONResponse(BUSY, status_code=503)
    except asyncio.CancelledError:
        inflight_finish(key, future, error="cancelled")
        raise
    try:
        result = await run_in_threadpool(answer_question, question, doc_ids, effort)
        cache_put(key, result)
        inflight_finish(key, future, result=result)
    except PageIndexAPIError as exc:
        metrics["errors"] += 1
        log.warning("retrieval failed: %s", exc)
        inflight_finish(key, future, error=str(exc))
        return JSONResponse({"error": RETRIEVAL_FAILED}, status_code=502)
    except Exception as exc:  # noqa: BLE001
        metrics["errors"] += 1
        log.exception("retrieval failed")
        inflight_finish(key, future, error=type(exc).__name__)
        return JSONResponse({"error": RETRIEVAL_FAILED}, status_code=502)
    finally:
        release_slot(started)
        if not future.done():
            # Cancelled mid-run: release anyone waiting on this key.
            inflight_finish(key, future, error="cancelled")

    return JSONResponse(result)


async def ask_stream(request: Request):
    started = time.monotonic()
    metrics["requests"] += 1
    try:
        body = await read_json(request)
        question, doc_ids, effort = parse_question(body)
    except RequestProblem as exc:
        metrics["errors"] += 1
        return JSONResponse({"error": str(exc)}, status_code=exc.status_code)

    key = cache_key(question, doc_ids, effort)
    cached = cache_get(key)
    if cached is not None:
        note_duration(started)
        return replay_result(cached)
    metrics["cache_misses"] += 1

    pending = inflight.get(key)
    if pending is not None:
        outcome = await inflight_wait(pending)
        if "__error__" in outcome:
            metrics["errors"] += 1
            return JSONResponse({"error": RETRIEVAL_FAILED}, status_code=502)
        note_duration(started)
        return replay_result(outcome)

    future = inflight_begin(key)
    try:
        if not await acquire_slot():
            inflight_finish(key, future, error="busy")
            return JSONResponse(BUSY, status_code=503)
    except asyncio.CancelledError:
        inflight_finish(key, future, error="cancelled")
        raise

    def start_stream():
        return CLIENT.chat(question, stream=True, show_process=False,
                           **chat_kwargs(doc_ids, effort))

    try:
        stream = await run_in_threadpool(start_stream)
    except asyncio.CancelledError:
        inflight_finish(key, future, error="cancelled")
        release_slot(started)
        raise
    except Exception as exc:  # noqa: BLE001 - the client never sees internals
        metrics["errors"] += 1
        log.exception("stream start failed")
        inflight_finish(key, future, error=type(exc).__name__)
        release_slot(started)
        return JSONResponse({"error": RETRIEVAL_FAILED}, status_code=502)

    async def frames() -> AsyncIterator[str]:
        answer, thinking = "", []
        try:
            async for event in iterate_in_threadpool(stream.events):
                kind = event.get("type")
                if kind == "answer":
                    delta = event.get("delta") or ""
                    if delta:
                        answer += delta
                        yield sse({"delta": delta})
                elif kind == "thinking":
                    delta = event.get("delta") or ""
                    if delta:
                        thinking.append(delta)
                        yield sse({"thinking": delta})
                elif kind == "tool_call":
                    narration = answer
                    answer = ""
                    if narration.strip():
                        thinking.append(narration)
                        yield sse({"to_thinking": narration})
                    yield sse({"tool": event.get("name") or ""})

            citations = await run_in_threadpool(read_citations, answer)
            result = {"answer": answer,
                      "thinking": "\n\n".join(p.strip() for p in thinking if p.strip()),
                      "citations": citations}
            cache_put(key, result)
            inflight_finish(key, future, result=result)
            yield sse({**result, "done": True})
        except PageIndexAPIError as exc:
            metrics["errors"] += 1
            log.warning("stream failed: %s", exc)
            inflight_finish(key, future, error=str(exc))
            yield sse({"error": RETRIEVAL_FAILED})
        except Exception as exc:  # noqa: BLE001
            metrics["errors"] += 1
            log.exception("stream failed")
            inflight_finish(key, future, error=type(exc).__name__)
            yield sse({"error": RETRIEVAL_FAILED})
        finally:
            if not future.done():
                inflight_finish(key, future, error="stream ended")
            stream.close()
            release_slot(started)

    return StreamingResponse(frames(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


async def internal_error(_request: Request, exc: Exception) -> JSONResponse:
    log.error("unhandled error: %s\n%s", exc, traceback.format_exc())
    metrics["errors"] += 1
    return JSONResponse({"error": "Internal server error."}, status_code=500)


routes = [
    Route("/", home),
    Route("/health", health),
    Route("/ready", ready),
    Route("/metrics", metrics_endpoint),
    Route("/api/documents", documents),
    Route("/api/ask", ask, methods=["POST"]),
    Route("/api/ask/stream", ask_stream, methods=["POST"]),
]

app = Starlette(
    routes=routes,
    middleware=[Middleware(OperationsMiddleware)],
    exception_handlers={Exception: internal_error},
)


if __name__ == "__main__":
    log.info("starting host=%s port=%s workers=%s model=%s store=%s",
             HOST, PORT, WORKERS, MODEL, STORE)
    uvicorn.run("server:app", host=HOST, port=PORT, workers=WORKERS,
                log_config=None, access_log=True)
