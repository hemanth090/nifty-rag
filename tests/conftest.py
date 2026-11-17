"""Test configuration: make the repo importable and keep the client offline.

The server module builds a PageIndex client at import. These environment values
give it a dummy key and an unreachable base URL; no test calls the provider, and
the fake client in test_server.py replaces every method that would.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("OPENCODE_GO_API_KEY", "test-key")
os.environ.setdefault("OPENCODE_GO_BASE_URL", "https://example.invalid/v1")
os.environ.setdefault("PAGEINDEX_MODEL", "test-model")
os.environ.setdefault("PAGEINDEX_STORE", str(ROOT / ".pageindex"))
os.environ.setdefault("API_TOKEN", "")
