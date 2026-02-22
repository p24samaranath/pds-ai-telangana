"""
PDS Optimization System — FastAPI Application
AI-Powered Fair Price Shop (FPS) Optimization & Fraud Detection
"""
import os
import sys
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routes.health import router as health_router
from routes.agents import router as agents_router
from routes.forecasts import router as forecasts_router
from routes.fraud_alerts import router as fraud_router
from routes.geospatial import router as geo_router
from routes.dashboard import router as dashboard_router
from routes.data import router as data_router
from app.config import settings

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info("All agents initialised and ready")
    yield
    logger.info("Shutting down PDS Optimization System")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Multi-agent AI system for India's Public Distribution System (PDS). "
        "Provides demand forecasting, fraud detection, geospatial optimization, "
        "and intelligent reporting for Fair Price Shops (FPS) in Telangana."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow React frontend (dev) and GitHub Pages (prod)
# NOTE: allow_credentials=True is incompatible with allow_origins=["*"]
_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://samaranathreddymukka.github.io",  # GitHub Pages deployment
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health_router)
app.include_router(data_router)
app.include_router(agents_router)
app.include_router(forecasts_router)
app.include_router(fraud_router)
app.include_router(geo_router)
app.include_router(dashboard_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
