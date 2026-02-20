#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# PDS AI System — Production Launcher
# Kills any stale sessions, starts backend + frontend, waits for both to be
# healthy, then opens the browser. Traps SIGINT/SIGTERM for clean shutdown.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_PORT=8000
FRONTEND_PORT=3000
BACKEND_PID_FILE="$SCRIPT_DIR/.backend.pid"
FRONTEND_PID_FILE="$SCRIPT_DIR/.frontend.pid"
LOG_DIR="$SCRIPT_DIR/backend/logs"
BACKEND_LOG="$LOG_DIR/backend.log"
FRONTEND_LOG="$LOG_DIR/frontend.log"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

log()  { echo -e "${CYAN}[$(date '+%H:%M:%S')]${RESET} $*"; }
ok()   { echo -e "${GREEN}✅ $*${RESET}"; }
warn() { echo -e "${YELLOW}⚠️  $*${RESET}"; }
err()  { echo -e "${RED}❌ $*${RESET}"; }
banner(){ echo -e "\n${BOLD}${BLUE}$*${RESET}\n"; }

# ── Step 1: Kill every previous session ───────────────────────────────────────
kill_previous_sessions() {
    banner "── Step 1/4: Cleaning up previous sessions ──"

    # Kill by PID file
    for pidfile in "$BACKEND_PID_FILE" "$FRONTEND_PID_FILE"; do
        if [ -f "$pidfile" ]; then
            OLD_PID=$(cat "$pidfile" 2>/dev/null || true)
            if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
                log "Killing saved PID $OLD_PID"
                kill -TERM "$OLD_PID" 2>/dev/null || true
                sleep 1
                kill -KILL "$OLD_PID" 2>/dev/null || true
            fi
            rm -f "$pidfile"
        fi
    done

    # Kill any process holding the ports
    for port in $BACKEND_PORT $FRONTEND_PORT; do
        PIDS=$(lsof -ti tcp:"$port" 2>/dev/null || true)
        if [ -n "$PIDS" ]; then
            log "Freeing port $port (PIDs: $PIDS)"
            echo "$PIDS" | xargs kill -TERM 2>/dev/null || true
            sleep 1
            echo "$PIDS" | xargs kill -KILL 2>/dev/null || true
        fi
    done

    # Kill lingering uvicorn / vite processes by name
    pkill -f "uvicorn app.main:app" 2>/dev/null || true
    pkill -f "vite"                 2>/dev/null || true
    sleep 1
    ok "Previous sessions cleared"
}

# ── Step 2: Preflight checks ──────────────────────────────────────────────────
preflight_checks() {
    banner "── Step 2/4: Preflight checks ──"

    # Python venv
    VENV="$SCRIPT_DIR/backend/venv"
    if [ ! -f "$VENV/bin/activate" ]; then
        err "Python venv not found at $VENV"
        err "Run: cd backend && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
        exit 1
    fi
    ok "Python venv found"

    # Key packages inside venv
    source "$VENV/bin/activate"
    python -c "import fastapi, uvicorn, anthropic, sklearn, torch" 2>/dev/null \
        && ok "Python packages OK" \
        || { err "Missing Python packages. Run: cd backend && pip install -r requirements.txt"; exit 1; }

    # Node / npm
    NPM_BIN=$(command -v npm 2>/dev/null || ls /opt/homebrew/bin/npm 2>/dev/null || true)
    if [ -z "$NPM_BIN" ]; then
        err "npm not found. Run: brew install node"
        exit 1
    fi
    NODE_MODULES="$SCRIPT_DIR/frontend/node_modules"
    if [ ! -d "$NODE_MODULES" ]; then
        warn "node_modules missing — installing…"
        "$NPM_BIN" --prefix "$SCRIPT_DIR/frontend" install --silent
    fi
    ok "Node/npm OK"

    # Data directory & env file
    mkdir -p "$SCRIPT_DIR/backend/data/raw" "$LOG_DIR"
    [ -f "$SCRIPT_DIR/.env" ] || { cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env" 2>/dev/null; \
        warn ".env created from .env.example — set ANTHROPIC_API_KEY for AI features"; }
    ok "Data directories & .env OK"
}

# ── Step 3: Start services ─────────────────────────────────────────────────────
start_services() {
    banner "── Step 3/4: Starting services ──"
    source "$SCRIPT_DIR/backend/venv/bin/activate"

    # Backend
    log "Starting backend on port $BACKEND_PORT…"
    cd "$SCRIPT_DIR/backend"
    PYTHONPATH=. nohup python -m uvicorn app.main:app \
        --host 0.0.0.0 --port $BACKEND_PORT \
        --log-level info \
        > "$BACKEND_LOG" 2>&1 &
    BACKEND_PID=$!
    echo "$BACKEND_PID" > "$BACKEND_PID_FILE"
    log "Backend PID: $BACKEND_PID"

    # Frontend
    log "Starting frontend on port $FRONTEND_PORT…"
    NPM_BIN=$(command -v npm 2>/dev/null || echo /opt/homebrew/bin/npm)
    cd "$SCRIPT_DIR/frontend"
    nohup "$NPM_BIN" run dev -- --port $FRONTEND_PORT \
        > "$FRONTEND_LOG" 2>&1 &
    FRONTEND_PID=$!
    echo "$FRONTEND_PID" > "$FRONTEND_PID_FILE"
    log "Frontend PID: $FRONTEND_PID"
    cd "$SCRIPT_DIR"
}

# ── Health polling ─────────────────────────────────────────────────────────────
wait_for_service() {
    local url=$1 label=$2 max_wait=${3:-60}
    log "Waiting for $label…"
    local elapsed=0
    until curl -sf "$url" >/dev/null 2>&1; do
        sleep 2; elapsed=$((elapsed+2))
        if [ $elapsed -ge $max_wait ]; then
            err "$label did not become ready in ${max_wait}s"
            err "Check log: $LOG_DIR"
            return 1
        fi
        printf "."
    done
    echo ""
    ok "$label is ready"
}

# ── Step 4: Open browser & wait ───────────────────────────────────────────────
open_and_wait() {
    banner "── Step 4/4: Opening application ──"
    wait_for_service "http://localhost:$BACKEND_PORT/api/v1/health" "Backend API"
    wait_for_service "http://localhost:$FRONTEND_PORT" "Frontend UI" 90

    ok "PDS AI System is running!"
    echo -e "\n  ${BOLD}Dashboard${RESET}  → http://localhost:$FRONTEND_PORT"
    echo -e "  ${BOLD}API Docs${RESET}   → http://localhost:$BACKEND_PORT/docs"
    echo -e "  ${BOLD}Backend log${RESET} → $BACKEND_LOG"
    echo -e "  ${BOLD}Frontend log${RESET} → $FRONTEND_LOG"
    echo -e "\n  ${YELLOW}Press Ctrl+C or click the EXIT button in the UI to stop.${RESET}\n"

    # Open browser on macOS
    open "http://localhost:$FRONTEND_PORT" 2>/dev/null || \
        xdg-open "http://localhost:$FRONTEND_PORT" 2>/dev/null || true

    # Wait forever until user stops
    wait
}

# ── Clean shutdown ─────────────────────────────────────────────────────────────
shutdown() {
    echo ""
    banner "── Shutting down PDS AI System ──"

    for pidfile in "$BACKEND_PID_FILE" "$FRONTEND_PID_FILE"; do
        if [ -f "$pidfile" ]; then
            PID=$(cat "$pidfile" 2>/dev/null || true)
            if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
                log "Stopping PID $PID"
                kill -TERM "$PID" 2>/dev/null || true
            fi
            rm -f "$pidfile"
        fi
    done

    # Belt-and-suspenders: kill by port
    for port in $BACKEND_PORT $FRONTEND_PORT; do
        lsof -ti tcp:"$port" 2>/dev/null | xargs kill -TERM 2>/dev/null || true
    done

    ok "PDS AI System stopped cleanly."
    exit 0
}

trap shutdown SIGINT SIGTERM EXIT

# ── Main ──────────────────────────────────────────────────────────────────────
echo -e "\n${BOLD}${GREEN}🌾 PDS AI Optimization System${RESET}"
echo -e "${CYAN}   Multi-Agent AI for Telangana's Public Distribution System${RESET}\n"

kill_previous_sessions
preflight_checks
start_services
open_and_wait
