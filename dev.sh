#!/usr/bin/env bash
set -euo pipefail

SESSION="sermon-translate"
ROOT="$(cd "$(dirname "$0")" && pwd)"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    exec tmux attach-session -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" -c "$ROOT/server" -n dev
tmux send-keys -t "$SESSION" "uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000" Enter
tmux split-window -h -t "$SESSION" -c "$ROOT/client"
tmux send-keys -t "$SESSION" "pnpm dev" Enter
tmux select-pane -t "$SESSION:.0"
exec tmux attach-session -t "$SESSION"
