#!/usr/bin/env bash
# Convenience launcher: creates a venv, installs deps, starts the server.
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  python3 -m venv .venv
  ./.venv/bin/pip install --upgrade pip
  ./.venv/bin/pip install -r requirements.txt
fi

export HF_HOME="${HF_HOME:-$PWD/model_cache}"
exec ./.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
