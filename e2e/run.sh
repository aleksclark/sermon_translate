#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> Building client dist..."
cd "$ROOT/client" && pnpm build

echo "==> Running e2e tests..."
cd "$ROOT/e2e"
docker compose up --build --abort-on-container-exit --exit-code-from playwright
docker compose down
