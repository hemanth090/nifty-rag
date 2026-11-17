"""Indexer tests: environment-driven budget defaults and backend wiring.

These exercise argument parsing and backend construction only; no indexing run
and no provider call happens.
"""

from __future__ import annotations

import pytest

import index_pdfs

ENV_NAMES = (
    "INDEX_MODE", "INDEX_CONCURRENCY", "INDEX_OPTIMIZE", "INDEX_PAUSE",
    "INDEX_SUMMARY_MAX_WORDS", "PAGEINDEX_MODEL", "PAGEINDEX_STORE",
    "INDEX_JSON_DIR",
)


@pytest.fixture()
def clean_env(monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_parser_uses_default_budgets(clean_env):
    args = index_pdfs.build_parser().parse_args([])
    assert args.mode == "flash"
    assert args.concurrency == 16
    assert args.optimize == "merge"
    assert args.pause == 2.0
    assert args.summary_max_words is None
    assert args.model == "deepseek-v4.1-flash"


def test_parser_reads_environment_budgets(clean_env):
    clean_env.setenv("INDEX_MODE", "standard")
    clean_env.setenv("INDEX_CONCURRENCY", "4")
    clean_env.setenv("INDEX_OPTIMIZE", "full")
    clean_env.setenv("INDEX_PAUSE", "0.5")
    clean_env.setenv("INDEX_SUMMARY_MAX_WORDS", "90")
    clean_env.setenv("PAGEINDEX_MODEL", "test-model")
    args = index_pdfs.build_parser().parse_args([])
    assert args.mode == "standard"
    assert args.concurrency == 4
    assert args.optimize == "full"
    assert args.pause == 0.5
    assert args.summary_max_words == 90
    assert args.model == "test-model"


def test_cli_flags_override_environment(clean_env):
    clean_env.setenv("INDEX_CONCURRENCY", "4")
    clean_env.setenv("INDEX_OPTIMIZE", "merge")
    args = index_pdfs.build_parser().parse_args(
        ["--concurrency", "128", "--optimize", "off",
         "--summary-max-words", "200", "--pause", "0"])
    assert args.concurrency == 128
    assert args.optimize == "off"
    assert args.summary_max_words == 200
    assert args.pause == 0


def test_invalid_environment_value_is_rejected(clean_env):
    clean_env.setenv("INDEX_OPTIMIZE", "bogus")
    with pytest.raises(SystemExit):
        index_pdfs.build_parser()


def test_invalid_concurrency_environment_is_rejected(clean_env):
    clean_env.setenv("INDEX_CONCURRENCY", "many")
    with pytest.raises(SystemExit):
        index_pdfs.build_parser()


def test_summary_max_words_must_be_positive(clean_env):
    clean_env.setenv("INDEX_SUMMARY_MAX_WORDS", "0")
    with pytest.raises(SystemExit):
        index_pdfs.build_parser()


def test_backend_reasoning_effort_comes_from_environment(clean_env):
    clean_env.setenv("OPENCODE_GO_API_KEY", "test-key")
    clean_env.setenv("INDEX_REASONING_EFFORT", "low")
    clean_env.delenv("OPENCODE_GO_BASE_URL", raising=False)
    backend = index_pdfs.opencode_backend()
    assert backend["api_key"] == "test-key"
    assert backend["api_base"] == index_pdfs.BASE_URL
    assert backend["reasoning_effort"] == "low"
    assert backend["extra_headers"]["User-Agent"].startswith("nifty-rag/")
