"""Server tests: routes, validation, cache, auth, streaming, and failure paths.

Every test replaces the PageIndex client methods with fakes, so the suite runs
offline and never mutates the real store or calls the provider.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

import server

DOCUMENTS = [
    {"id": "pi-1", "name": "TCS_FY25.pdf", "pageNum": 535},
    {"id": "pi-2", "name": "INFY_FY25.pdf", "pageNum": 320},
]

ANSWER = 'TCS revenue was 255,324 crore. <cite doc="TCS_FY25.pdf" page="10"/>'


class FakeStream:
    def __init__(self) -> None:
        self.events = [
            {"type": "thinking", "delta": "reading the tree"},
            {"type": "tool_call", "name": "get_page_content"},
            {"type": "answer", "delta": "TCS revenue "},
            {"type": "answer", "delta": "was 255,324 crore."},
        ]
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _install_fake_client(monkeypatch, calls: list[dict]) -> None:
    def list_documents(limit: int = 10000, **_kwargs):
        return {"documents": list(DOCUMENTS), "total": len(DOCUMENTS)}

    def chat(question: str, **kwargs):
        calls.append({"question": question, **kwargs})
        if kwargs.get("stream"):
            return FakeStream()
        return ANSWER

    def get_citations(answer: str):
        return [{"document": "TCS_FY25.pdf", "page": 10, "doc_id": "pi-1"}]

    monkeypatch.setattr(server.CLIENT, "list_documents", list_documents)
    monkeypatch.setattr(server.CLIENT, "chat", chat)
    monkeypatch.setattr(server.CLIENT, "get_citations", get_citations)


@pytest.fixture()
def http(monkeypatch):
    """A TestClient with a fake PageIndex client and reset runtime state."""
    server.cache.clear()
    server.catalog_cache.update({"signature": "", "text": ""})
    for key in ("requests", "errors", "busy", "cache_hits", "cache_misses", "active"):
        server.metrics[key] = 0
    server.metrics["durations_ms"].clear()
    calls: list[dict] = []
    _install_fake_client(monkeypatch, calls)
    with TestClient(server.app) as client:
        yield client, calls


# ── Health and catalog ───────────────────────────────────────────────────

def test_health(http):
    client, _ = http
    body = client.get("/health").json()
    assert body["ok"] is True
    assert body["version"] == server.APP_VERSION
    assert body["model"] == server.MODEL
    assert body["uptime_s"] >= 0
    budgets = body["budgets"]
    assert budgets["max_turns"] == server.MAX_TURNS
    assert budgets["max_output_tokens"] == server.MAX_OUTPUT_TOKENS
    assert budgets["tool_response_chars"] == server.TOOL_RESPONSE_CHARS
    assert budgets["tool_max_pages"] == server.TOOL_MAX_PAGES
    assert budgets["catalog_chars"] == server.CATALOG_CHARS


def test_ready_reports_store(http):
    client, _ = http
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["ready"] is True


def test_metrics_fields(http):
    client, _ = http
    body = client.get("/metrics").json()
    for field in ("requests", "errors", "cache_hits", "cache_misses",
                  "cache_size", "active", "max_concurrent", "model", "uptime_s",
                  "budgets"):
        assert field in body
    assert body["budgets"]["max_concurrent"] == server.MAX_CONCURRENT
    assert body["budgets"]["max_turns"] == server.MAX_TURNS
    assert body["budgets"]["max_output_tokens"] == server.MAX_OUTPUT_TOKENS
    assert body["budgets"]["cache_ttl_s"] == server.CACHE_TTL
    assert body["budgets"]["cache_max"] == server.CACHE_MAX


def test_documents(http):
    client, _ = http
    body = client.get("/api/documents").json()
    assert [doc["name"] for doc in body["documents"]] == [
        "TCS_FY25.pdf", "INFY_FY25.pdf"]
    assert body["documents"][0]["pages"] == 535


# ── Ask ──────────────────────────────────────────────────────────────────

def test_ask_returns_answer_and_citations(http):
    client, calls = http
    response = client.post("/api/ask", json={"question": "What was TCS revenue?"})
    assert response.status_code == 200
    body = response.json()
    assert "255,324 crore" in body["answer"]
    assert body["citations"] == [
        {"document": "TCS_FY25.pdf", "page": 10, "doc_id": "pi-1"}]
    assert calls[0]["citations"] is True
    assert calls[0]["max_turns"] == 10
    assert calls[0]["doc_id"] is None
    assert "TCS_FY25.pdf" in (calls[0]["instructions"] or "")


def test_ask_passes_document_scope(http):
    client, calls = http
    response = client.post("/api/ask",
                           json={"question": "Revenue?", "doc_ids": ["pi-1"]})
    assert response.status_code == 200
    assert calls[0]["doc_id"] == ["pi-1"]


def test_budgets_reach_the_chat_call(http, monkeypatch):
    client, calls = http
    monkeypatch.setattr(server, "MAX_TURNS", 7)
    monkeypatch.setattr(server, "MAX_OUTPUT_TOKENS", 1234)
    response = client.post("/api/ask", json={"question": "Budget check?"})
    assert response.status_code == 200
    assert calls[0]["max_turns"] == 7
    assert calls[0]["extra_body"] == {"max_tokens": 1234}


def test_zero_output_budget_leaves_provider_default(http, monkeypatch):
    client, calls = http
    monkeypatch.setattr(server, "MAX_OUTPUT_TOKENS", 0)
    response = client.post("/api/ask", json={"question": "No output cap?"})
    assert response.status_code == 200
    assert "extra_body" not in calls[0]


def test_tool_budgets_are_applied_to_the_sdk():
    assert server.agent_tools.TOOL_RESPONSE_CHAR_LIMIT == server.TOOL_RESPONSE_CHARS
    assert server.agent_tools._CHAR_BUDGET == int(server.TOOL_RESPONSE_CHARS * 0.95)
    assert server.agent_tools._MAX_REQUESTED_PAGES == server.TOOL_MAX_PAGES
    assert server.agent_tools._SIMILAR_NAMES_LIMIT == server.TOOL_SUGGESTIONS


def test_ask_rejects_empty_question(http):
    client, _ = http
    response = client.post("/api/ask", json={"question": "   "})
    assert response.status_code == 400
    assert "empty" in response.json()["error"]


def test_ask_rejects_long_question(http):
    client, _ = http
    response = client.post("/api/ask", json={"question": "x" * 1001})
    assert response.status_code == 400
    assert "longer than" in response.json()["error"]


def test_ask_rejects_unknown_effort(http):
    client, _ = http
    response = client.post("/api/ask",
                           json={"question": "q", "reasoning_effort": "extreme"})
    assert response.status_code == 400
    assert "reasoning_effort" in response.json()["error"]


def test_ask_rejects_bad_doc_ids(http):
    client, _ = http
    response = client.post("/api/ask", json={"question": "q", "doc_ids": "pi-1"})
    assert response.status_code == 400
    too_many = client.post("/api/ask",
                           json={"question": "q", "doc_ids": [f"pi-{i}" for i in range(65)]})
    assert too_many.status_code == 400


def test_ask_rejects_non_object_json(http):
    client, _ = http
    response = client.post("/api/ask", content=b"[1, 2]",
                           headers={"content-type": "application/json"})
    assert response.status_code == 400
    assert "JSON object" in response.json()["error"]


def test_ask_rejects_wrong_content_type(http):
    client, _ = http
    response = client.post("/api/ask", content=b'{"question": "q"}',
                           headers={"content-type": "text/plain"})
    assert response.status_code == 415


def test_ask_rejects_oversized_body(http, monkeypatch):
    client, _ = http
    monkeypatch.setattr(server, "MAX_BODY_BYTES", 10)
    response = client.post("/api/ask", json={"question": "what is this?"})
    assert response.status_code == 413


def test_ask_caches_answers(http):
    client, calls = http
    first = client.post("/api/ask", json={"question": "What was TCS revenue?"})
    second = client.post("/api/ask", json={"question": "What was TCS revenue?"})
    assert first.status_code == second.status_code == 200
    assert second.json()["cached"] is True
    assert len(calls) == 1
    assert server.metrics["cache_hits"] == 1


def test_ask_returns_503_when_busy(http, monkeypatch):
    client, _ = http

    async def busy():
        return False

    monkeypatch.setattr(server, "acquire_slot", busy)
    response = client.post("/api/ask", json={"question": "What was TCS revenue?"})
    assert response.status_code == 503
    assert "too many questions" in response.json()["error"]


def test_citations_failure_keeps_the_answer(http, monkeypatch):
    client, _ = http

    def broken(_answer):
        raise RuntimeError("citation parser is down")

    monkeypatch.setattr(server.CLIENT, "get_citations", broken)
    response = client.post("/api/ask", json={"question": "What was TCS revenue?"})
    assert response.status_code == 200
    assert response.json()["citations"] == []
    assert "255,324 crore" in response.json()["answer"]


# ── Streaming ────────────────────────────────────────────────────────────

def test_stream_emits_frames_and_done(http):
    client, _ = http
    response = client.post("/api/ask/stream",
                           json={"question": "What was TCS revenue?"})
    assert response.status_code == 200
    text = response.text
    assert "data: " in text
    assert '"delta": "TCS revenue "' in text
    assert '"thinking": "reading the tree"' in text
    assert '"tool": "get_page_content"' in text
    assert '"done": true' in text
    assert '"citations"' in text


def test_stream_replays_cache(http):
    client, calls = http
    client.post("/api/ask/stream", json={"question": "What was TCS revenue?"})
    replay = client.post("/api/ask/stream",
                         json={"question": "What was TCS revenue?"})
    assert replay.status_code == 200
    assert '"delta": "TCS revenue was 255,324 crore."' in replay.text
    assert '"done": true' in replay.text
    assert len(calls) == 1


# ── Auth ─────────────────────────────────────────────────────────────────

def test_bearer_token_protects_api(http, monkeypatch):
    client, _ = http
    monkeypatch.setattr(server, "API_TOKEN", "secret")
    assert client.get("/api/documents").status_code == 401
    assert client.get("/metrics").status_code == 401
    assert client.get("/health").status_code == 200
    allowed = client.get("/api/documents",
                         headers={"Authorization": "Bearer secret"})
    assert allowed.status_code == 200
