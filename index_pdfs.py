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

Budgets come from the environment, then CLI flags override them:

    INDEX_CONCURRENCY=16        parallel summary calls per PDF
    INDEX_OPTIMIZE=merge        full | merge | off
    INDEX_PAUSE=2               seconds between PDFs
    INDEX_SUMMARY_MAX_WORDS=150 word cap per node summary
    INDEX_REASONING_EFFORT=none reasoning effort for summary/expand calls
    INDEX_MODE=flash            flash | standard
    PAGEINDEX_MODEL=deepseek-v4.1-flash
    PAGEINDEX_STORE=./.pageindex
    INDEX_JSON_DIR=./results

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


def _env_int(name: str, default: int, minimum: int | None = None) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        raise SystemExit(f"{name} must be an integer, got {raw!r}.")
    if minimum is not None and value < minimum:
        raise SystemExit(f"{name} must be >= {minimum}, got {value}.")
    return value


def _env_opt_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError:
        raise SystemExit(f"{name} must be an integer, got {raw!r}.")
    if value < 1:
        raise SystemExit(f"{name} must be >= 1, got {value}.")
    return value


def _env_float(name: str, default: float, minimum: float | None = None) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        raise SystemExit(f"{name} must be a number, got {raw!r}.")
    if minimum is not None and value < minimum:
        raise SystemExit(f"{name} must be >= {minimum}, got {value}.")
    return value


def _env_choice(name: str, default: str, choices: tuple[str, ...]) -> str:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value not in choices:
        raise SystemExit(f"{name} must be one of {list(choices)}, got {raw!r}.")
    return value


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return raw.strip() if raw and raw.strip() else default


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
        "reasoning_effort": _env_str("INDEX_REASONING_EFFORT", "none"),
    }


@contextmanager
def store_lock(path: Path | None = None):
    """One indexing run at a time against a store."""
    path = path or LOCK_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8")
    locked = False
    try:
        if fcntl is not None:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except BlockingIOError:
                raise SystemExit(
                    f"Another indexing run holds {path}. Stop it or wait.")
        handle.write(str(os.getpid()))
        handle.flush()
        yield
    finally:
        try:
            if fcntl is not None and locked:
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


def build_parser() -> argparse.ArgumentParser:
    """CLI parser whose defaults are the effective environment budgets."""
    model = _env_str("PAGEINDEX_MODEL", "deepseek-v4.1-flash")
    mode = _env_choice("INDEX_MODE", "flash", ("flash", "standard"))
    store = _env_str("PAGEINDEX_STORE", str(DEFAULT_STORE))
    json_dir = _env_str("INDEX_JSON_DIR", str(DEFAULT_JSON_DIR))
    concurrency = _env_int("INDEX_CONCURRENCY", 16, minimum=0)
    optimize = _env_choice("INDEX_OPTIMIZE", "merge", ("full", "merge", "off"))
    summary_words = _env_opt_int("INDEX_SUMMARY_MAX_WORDS")
    pause = _env_float("INDEX_PAUSE", 2.0, minimum=0.0)

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pdfs", nargs="*", help="PDF files (default: all of data/pdfs)")
    parser.add_argument("--model", default=model,
                        help=f"local index model (default: {model}; env PAGEINDEX_MODEL)")
    parser.add_argument("--mode", choices=["flash", "standard"], default=mode,
                        help=f"tree-building mode (default: {mode}; env INDEX_MODE)")
    parser.add_argument("--store", default=store,
                        help=f"local SDK storage directory (default: {store})")
    parser.add_argument("--json-dir", default=json_dir,
                        help="also write <name>_structure.json here; pass '' to disable "
                             f"(default: {json_dir})")
    parser.add_argument("--concurrency", type=int, default=concurrency,
                        help=f"parallel summary calls per PDF (default: {concurrency}; "
                             "0 uses PageIndex's 128, which triggers OpenCode Go 429s; "
                             "env INDEX_CONCURRENCY)")
    parser.add_argument("--optimize", choices=["full", "merge", "off"],
                        default=optimize,
                        help=f"tree refinement (default: {optimize}; full adds the LLM "
                             "expand pass; env INDEX_OPTIMIZE)")
    parser.add_argument("--summary-max-words", type=int, default=summary_words,
                        help="word cap per node summary (default: "
                             f"{summary_words if summary_words else 'PageIndex 150'}; "
                             "env INDEX_SUMMARY_MAX_WORDS)")
    parser.add_argument("--pause", type=float, default=pause,
                        help=f"seconds to wait between PDFs (default: {pause}; "
                             "env INDEX_PAUSE)")
    parser.add_argument("--force", action="store_true",
                        help="re-index documents already in the store")
    return parser


def main(argv=None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.concurrency < 0:
        parser.error("--concurrency must be 0 (SDK default) or a positive integer")
    if args.pause < 0:
        parser.error("--pause must be >= 0")
    if args.summary_max_words is not None and args.summary_max_words < 1:
        parser.error("--summary-max-words must be a positive integer")

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
    if args.summary_max_words:
        index_conf["summary_max_words"] = args.summary_max_words

    with store_lock():
        client = PageIndexClient(index=index_conf)
        existing: dict[str, list[str]] = {}
        for doc in client.list_documents(limit=10000)["documents"]:
            existing.setdefault(doc["name"], []).append(doc["id"])
        log.info("store=%s model=%s mode=%s optimize=%s concurrency=%s "
                 "summary_max_words=%s reasoning_effort=%s documents=%d pdfs=%d",
                 store, args.model, args.mode, args.optimize, args.concurrency,
                 args.summary_max_words or "sdk-default(150)",
                 _env_str("INDEX_REASONING_EFFORT", "none"),
                 len(existing), len(pdfs))

        totals = {"prompt": 0, "output": 0, "reasoning": 0}
        failures: list[tuple[str, str]] = []
        skipped = 0
        for number, pdf in enumerate(pdfs, 1):
            name = pdf.name
            if not pdf.is_file():
                failures.append((name, "file not found"))
                log.error("[%d/%d] FAIL %s: file not found", number, len(pdfs), name)
                continue
            ids = existing.get(name, [])
            if ids and not args.force:
                skipped += 1
                log.info("[%d/%d] skip %s (already in store)", number, len(pdfs), name)
                continue
            if ids and args.force:
                log.info("[%d/%d] replacing %s (%d existing document(s))",
                         number, len(pdfs), name, len(ids))
                replaced = True
                for doc_id in ids:
                    try:
                        client.delete_document(doc_id)
                    except Exception as exc:  # noqa: BLE001
                        replaced = False
                        failures.append((name, f"replace failed: {exc}"))
                        log.error("    could not delete %s for %s: %s", doc_id, name, exc)
                if not replaced:
                    continue
                existing.pop(name, None)
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
                    try:
                        client.delete_document(result["doc_id"])
                        log.info("    rolled back doc_id=%s so the next run retries it",
                                 result["doc_id"])
                    except Exception as rollback_exc:  # noqa: BLE001
                        log.error("    rollback failed for %s: %s",
                                  result["doc_id"], rollback_exc)
            if args.pause and number < len(pdfs):
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
