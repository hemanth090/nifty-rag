#!/usr/bin/env bash
# Production launcher: the repository's virtualenv, one or more uvicorn workers.
#
#   ./run_server.sh
#   HOST=0.0.0.0 PORT=8080 WORKERS=2 ./run_server.sh
#
# Settings come from .env (loaded by server.py) and the environment.
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/python server.py
