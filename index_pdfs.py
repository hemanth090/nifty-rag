#!/usr/bin/env python3
"""Index PDFs into the local PageIndex store through OpenCode Go (DeepSeek V4.1 Flash).

Runs on the stock PageIndex package: the provider is passed as the local SDK's
``index.backend``, which LiteLLM forwards to OpenCode Go. Nothing in this
script forks or patches the SDK.

Run from the repository root:

    .venv/bin/python index_pdfs.py                       # every data/pdfs/*.pdf
    .venv/bin/python index_pdfs.py data/pdfs/TCS_FY25.pdf --force
    .venv/bin/python index_pdfs.py --json-dir ''         # store only, no JSON

Writes:

    .pageindex/                     the local SDK store the server reads
    results/<name>_structure.json   the same tree as JSON, one per PDF

Skips documents already in the store by filename unless --force is passed.
Logs prompt/output/reasoning tokens for every PDF and refuses to start when
another indexing run holds the store lock.

The stock local SDK summarizes 128 nodes in parallel, which trips OpenCode
Go's rate limit (HTTP 429) on ~500-page filings. This script passes a lower
``summary_concurrency`` (default 16) through the SDK's index config, and
pauses between PDFs. It also defaults ``optimize="merge"``: the LLM expand
pass builds enormous prompts on 600+ page filings (one was 262k chars) and
the gateway answers those with connection errors, so merge keeps the tree
and summaries without that pass. Pass ``--optimize full`` to re-enable it.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv

try:  # POSIX only; the importer still works elsewhere
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")  # explicit: works whatever the current directory is

BASE_URL = "https://opencode.ai/inference/go/openai/v1"
DEFAULT_STORE = ROOT / ".pageindex"
DEFAULT_JSON_DIR = ROOT / "results"
DEFAULT_PDF_DIR = ROOT / "data" / "pdfs"
LOCK_PATH = ROOT / ".index.lock"

log = logging.getLogger("nifty_rag.index")


def configure_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )


def opencode_backend() -> dict:
    """LiteLLM connection overrides for OpenCode Go (key, endpoint, headers)."""
    key = (os.environ.get("OPENCODE_GO_API_KEY") or "").strip() or \
          (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        raise SystemExit(
            f"No key: put OPENCODE_GO_API_KEY=... in {ROOT / '.env'} "
            "or export it in the environment.")
    try:
        from pageindex._version import sdk_version
        version = sdk_version()
    except Exception:  # noqa: BLE001 - version metadata is optional
        version = "unknown"
    return {
        "api_key": key,
        "api_base": (os.environ.get("OPENCODE_GO_BASE_URL") or BASE_URL).rstrip("/"),
        "extra_headers": {
            "User-Agent": f"nifty-rag/1.0.0 pageindex/{version} (coding-agent)",
            "HTTP-Referer": "https://opencode.ai",
            "x-opencode-session": (os.environ.get("OPENCODE_SESSION_ID")
                                   or f"nifty-rag-{uuid.uuid4().hex}"),
        },
        # Turn hidden thinking off for summaries *and* the expand pass.
        "reasoning_effort": "none",
    }


@contextmanager
def store_lock(path: Path = LOCK_PATH):
    """One indexing run at a time against a store."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8")
    try:
        if fcntl is not None:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                raise SystemExit(
                    f"Another indexing run holds {path}. Stop it or wait.")
        handle.write(str(os.getpid()))
        handle.flush()
        yield
    finally:
        try:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def write_json_atomic(path: Path, payload: dict) -> None:
    """Write JSON so a crash never leaves a half-written result file."""
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _usage_line(usage: dict) -> str:
    completion = usage.get("completion_tokens_details") or {}
    reasoning = completion.get("reasoning_tokens", 0)
    return (f"tokens: prompt={usage.get('prompt_tokens', 0)} "
            f"output={usage.get('completion_tokens', 0)} reasoning={reasoning}")


def _is_rate_limit(exc: Exception) -> bool:
    low = str(exc).lower()
    return "rate limit" in low or "429" in low


def _is_transient(exc: Exception) -> bool:
    low = str(exc).lower()
    return any(marker in low for marker in
               ("connection error", "internalservererror", "timeout", "timed out",
                "apiconnection", "502", "503", "504"))


def main(argv=None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pdfs", nargs="*", help="PDF files (default: all of data/pdfs)")
    parser.add_argument("--model", default=os.environ.get("PAGEINDEX_MODEL",
                                                          "deepseek-v4.1-flash"),
                        help="local index model (default: deepseek-v4.1-flash)")
    parser.add_argument("--mode", choices=["flash", "standard"], default="flash",
                        help="tree-building mode (default: flash)")
    parser.add_argument("--store", default=str(DEFAULT_STORE),
                        help="local SDK storage directory (default: ./.pageindex)")
    parser.add_argument("--json-dir", default=str(DEFAULT_JSON_DIR),
                        help="also write <name>_structure.json here; pass '' to disable")
    parser.add_argument("--concurrency", type=int, default=16,
                        help="parallel summary calls per PDF (default: 16; the "
                             "stock SDK default of 128 triggers OpenCode Go 429s; "
                             "16 measured clean and ~4x faster)")
    parser.add_argument("--optimize", choices=["full", "merge", "off"], default="merge",
                        help="tree refinement: merge (default, no LLM expand), "
                             "full (+ LLM expand; huge prompts fail on 600+ page PDFs), off")
    parser.add_argument("--pause", type=float, default=2.0,
                        help="seconds to wait between PDFs (default: 2)")
    parser.add_argument("--force", action="store_true",
                        help="re-index documents already in the store")
    args = parser.parse_args(argv)

    if args.concurrency < 0:
        parser.error("--concurrency must be 0 (SDK default) or a positive integer")
    if args.pause < 0:
        parser.error("--pause must be >= 0")

    pdfs = [Path(p).expanduser() for p in args.pdfs] or sorted(
        Path(p) for p in glob.glob(str(DEFAULT_PDF_DIR / "*.pdf")))
    if not pdfs:
        raise SystemExit(f"No PDFs found (looked in {DEFAULT_PDF_DIR} and the arguments).")

    from pageindex import PageIndexClient

    store = Path(args.store).expanduser()
    json_dir = Path(args.json_dir).expanduser() if args.json_dir else None
    store.mkdir(parents=True, exist_ok=True)
    if json_dir:
        json_dir.mkdir(parents=True, exist_ok=True)

    index_conf = {"model": args.model,
                  "storage_path": str(store),
                  "backend": opencode_backend(),
                  "optimize": args.optimize}
    if args.concurrency and args.concurrency > 0:
        index_conf["summary_concurrency"] = args.concurrency

    with store_lock():
        client = PageIndexClient(index=index_conf)
        existing = {doc["name"] for doc in
                    client.list_documents(limit=10000)["documents"]}
        log.info("store=%s model=%s mode=%s optimize=%s concurrency=%s "
                 "documents=%d pdfs=%d", store, args.model, args.mode,
                 args.optimize, args.concurrency, len(existing), len(pdfs))

        totals = {"prompt": 0, "output": 0, "reasoning": 0}
        failures: list[tuple[str, str]] = []
        skipped = 0
        for number, pdf in enumerate(pdfs, 1):
            name = pdf.name
            if not pdf.is_file():
                failures.append((name, "file not found"))
                log.error("[%d/%d] FAIL %s: file not found", number, len(pdfs), name)
                continue
            if name in existing and not args.force:
                skipped += 1
                log.info("[%d/%d] skip %s (already in store)", number, len(pdfs), name)
                continue
            log.info("[%d/%d] indexing %s", number, len(pdfs), name)
            started = time.monotonic()
            result = usage = None
            for attempt in (1, 2, 3):
                try:
                    result = client.submit_document(str(pdf), wait=True, mode=args.mode)
                    usage = result.get("usage") or {}
                    break
                except Exception as exc:  # noqa: BLE001 - one bad PDF must not stop the batch
                    error = f"{type(exc).__name__}: {exc}"
                    if (_is_rate_limit(exc) or _is_transient(exc)) and attempt < 3:
                        cooldown = 45 if _is_rate_limit(exc) else 30 * attempt
                        log.warning(
                            "    %s; cooling down %ss, then retrying %s (attempt %d/3)",
                            "rate limited" if _is_rate_limit(exc) else "gateway error",
                            cooldown, name, attempt + 1)
                        time.sleep(cooldown)
                        continue
                    failures.append((name, error))
                    log.error("[%d/%d] FAIL %s: %s", number, len(pdfs), name, error)
                    break
            if result is None:
                continue

            completion = usage.get("completion_tokens_details") or {}
            totals["prompt"] += usage.get("prompt_tokens", 0)
            totals["output"] += usage.get("completion_tokens", 0)
            totals["reasoning"] += completion.get("reasoning_tokens", 0)
            log.info("    saved doc_id=%s  %s  (%.0fs)", result["doc_id"],
                     _usage_line(usage), time.monotonic() - started)

            if json_dir:
                try:
                    tree = client.get_document_structure(result["doc_id"])
                    out = json_dir / f"{pdf.stem}_structure.json"
                    write_json_atomic(out, {"doc_name": name,
                                            "doc_id": result["doc_id"],
                                            "usage": usage,
                                            "structure": tree})
                    log.info("    tree -> %s", out)
                except Exception as exc:  # noqa: BLE001 - store write already succeeded
                    failures.append((name, f"tree export failed: {exc}"))
                    log.error("    tree export failed for %s: %s", name, exc)
            if args.pause:
                time.sleep(args.pause)

    attempted = len(pdfs) - skipped
    log.info("done: indexed=%d skipped=%d failed=%d of %d PDFs; store=%s json=%s",
             attempted - len(failures), skipped, len(failures), len(pdfs),
             store, json_dir or "<off>")
    log.info("totals: prompt=%d output=%d reasoning=%d tokens",
             totals["prompt"], totals["output"], totals["reasoning"])
    if failures:
        log.error("failures:")
        for name, error in failures:
            log.error("  %s: %s", name, error)
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log.error("interrupted")
        raise SystemExit(130)
