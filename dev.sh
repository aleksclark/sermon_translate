#!/usr/bin/env bash
set -euo pipefail

SESSION="sermon-translate"
ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/../../repo/server/.venv"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    exec tmux attach-session -t "$SESSION"
fi

# Server pane
tmux new-session -d -s "$SESSION" -c "$ROOT/server" -n dev
tmux send-keys -t "$SESSION" \
    "PYTHONPATH=. $VENV/bin/python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000" Enter

# Client pane
tmux split-window -h -t "$SESSION" -c "$ROOT/client"
tmux send-keys -t "$SESSION" \
    "[ -d node_modules ] || pnpm install; pnpm dev --host 0.0.0.0" Enter

tmux select-pane -t "$SESSION:.0"
exec tmux attach-session -t "$SESSION"
