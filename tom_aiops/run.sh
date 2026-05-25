#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Load .env if it exists
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

python -m uvicorn tom_aiops.app:app --app-dir . --host 127.0.0.1 --port 8010
