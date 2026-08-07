#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

python_bin="${RAG_PYTHON:-$project_dir/.venv/bin/python}"
exec "$python_bin" -X utf8 -u webapp/server.py \
  --host "${RAG_HOST:-127.0.0.1}" \
  --port "${RAG_PORT:-8765}"
