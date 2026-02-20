#!/bin/bash
# Clean shutdown — callable from UI or terminal
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GREEN='\033[0;32m'; RESET='\033[0m'

echo -e "\n🛑 Stopping PDS AI System…"

for pidfile in "$SCRIPT_DIR/.backend.pid" "$SCRIPT_DIR/.frontend.pid"; do
    if [ -f "$pidfile" ]; then
        PID=$(cat "$pidfile" 2>/dev/null || true)
        [ -n "$PID" ] && kill -TERM "$PID" 2>/dev/null || true
        rm -f "$pidfile"
    fi
done

for port in 8000 3000; do
    lsof -ti tcp:"$port" 2>/dev/null | xargs kill -TERM 2>/dev/null || true
done

pkill -f "uvicorn app.main:app" 2>/dev/null || true
pkill -f "vite" 2>/dev/null || true

sleep 1
echo -e "${GREEN}✅ PDS AI System stopped.${RESET}"
