"""Health check and system control routes."""
import os
import signal
import subprocess
import logging
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/v1", tags=["Health"])
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parents[3]   # repo root


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "PDS Optimization System",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/")
async def root():
    return {
        "message": "PDS AI Optimization System API",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


@router.post("/system/shutdown")
async def shutdown():
    """
    Graceful shutdown — called by the frontend Exit button.
    Runs stop.sh (which kills backend + frontend by PID / port),
    then sends SIGTERM to ourselves.
    """
    logger.info("Shutdown requested via API — running stop.sh")
    stop_script = SCRIPT_DIR / "stop.sh"
    if stop_script.exists():
        try:
            subprocess.Popen(
                ["bash", str(stop_script)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            logger.warning(f"stop.sh failed: {e}")

    # Give the HTTP response time to be sent, then kill self
    import asyncio
    async def _kill():
        await asyncio.sleep(0.5)
        os.kill(os.getpid(), signal.SIGTERM)

    asyncio.create_task(_kill())
    return JSONResponse(
        content={"status": "shutting_down", "message": "PDS system is shutting down…"},
        status_code=200,
    )
