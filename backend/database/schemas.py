from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from app.constants import (
    CardType, CommodityType, FraudSeverity, FraudPattern,
    TransactionStatus, AlertStatus
)


# ── FPS Shop ──────────────────────────────────────────────────────────────────

class FPSShopBase(BaseModel):
    shop_id: str
    shop_name: str
    dealer_name: Optional[str] = None
    district: str
    mandal: Optional[str] = None
    village: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    is_active: bool = True


class FPSShopCreate(FPSShopBase):
    pass


class FPSShopResponse(FPSShopBase):
    id: int
    total_cards: int
    created_at: datetime

    class Config:
        from_attributes = True


# ── Beneficiary ───────────────────────────────────────────────────────────────

class BeneficiaryBase(BaseModel):
    card_id: str
    card_type: CardType
    district: str
    members_count: int = 1
    is_active: bool = True


class BeneficiaryCreate(BeneficiaryBase):
    head_of_family: Optional[str] = None
    mobile_number: Optional[str] = None


class BeneficiaryResponse(BeneficiaryBase):
    id: int
    fps_shop_id: Optional[int] = None
    created_at: datetime

    class Config:
        from_attributes = True


# ── Transaction ───────────────────────────────────────────────────────────────

class TransactionBase(BaseModel):
    transaction_id: str
    commodity: CommodityType
    quantity_kg: float
    transaction_date: datetime
    status: TransactionStatus = TransactionStatus.COMPLETED
    biometric_verified: bool = False


class TransactionCreate(TransactionBase):
    beneficiary_id: int
    fps_shop_id: int


class TransactionResponse(TransactionBase):
    id: int
    anomaly_score: float
    created_at: datetime

    class Config:
        from_attributes = True


# ── Fraud Alert ───────────────────────────────────────────────────────────────

class FraudAlertResponse(BaseModel):
    id: int
    alert_id: str
    beneficiary_card_id: Optional[str]
    severity: FraudSeverity
    fraud_pattern: Optional[str]
    anomaly_score: float
    description: str
    explanation: Optional[str]
    recommended_action: Optional[str]
    status: AlertStatus
    detected_at: datetime

    class Config:
        from_attributes = True


# ── Demand Forecast ───────────────────────────────────────────────────────────

class ForecastRequest(BaseModel):
    shop_id: Optional[str] = None
    district: Optional[str] = None
    commodity: Optional[CommodityType] = None
    months_ahead: int = Field(default=3, ge=1, le=12)


class ForecastResult(BaseModel):
    fps_shop_id: Any        # str or int depending on source
    shop_name: str
    district: str
    commodity: CommodityType
    forecast_month: str
    predicted_quantity_kg: float
    confidence_lower: float
    confidence_upper: float
    model_used: str
    risk_flag: Optional[str] = None


class ForecastResponse(BaseModel):
    forecasts: List[ForecastResult]
    generated_at: datetime
    total_shops: int


# ── Geospatial ────────────────────────────────────────────────────────────────

class UnderservedZone(BaseModel):
    district: str
    village: Optional[str]
    latitude: float
    longitude: float
    nearest_fps_distance_km: float
    affected_beneficiaries: int
    priority_score: float


class NewLocationRecommendation(BaseModel):
    rank: int
    recommended_lat: float
    recommended_lon: float
    district: str
    projected_coverage: int
    current_avg_distance_km: float
    projected_avg_distance_km: float
    justification: str


class GeospatialResponse(BaseModel):
    underserved_zones: List[UnderservedZone]
    new_location_recommendations: List[NewLocationRecommendation]
    district_accessibility_scores: Dict[str, float]
    analysis_date: datetime


# ── Agent & Orchestrator ──────────────────────────────────────────────────────

class AgentQueryRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None
    agent_hint: Optional[str] = None
    session_id: str = "default"


class AgentQueryResponse(BaseModel):
    query: str
    answer: str
    agent_used: str
    data: Optional[Dict[str, Any]] = None
    generated_at: datetime


# ── RAG Chat ──────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    query: str
    answer: str
    session_id: str
    conversation_turn: int
    rag_sources: List[str]
    agent_used: str
    generated_at: datetime


class ChatSessionInfo(BaseModel):
    session_id: str
    turns: int
    messages: int


class OrchestratorStatus(BaseModel):
    status: str
    active_agents: List[str]
    last_fraud_check: Optional[datetime]
    last_forecast_run: Optional[datetime]
    last_geo_analysis: Optional[datetime]
    pending_alerts: int
    system_health: str


# ── Dashboard ─────────────────────────────────────────────────────────────────

class DashboardMetrics(BaseModel):
    total_fps_shops: int
    active_fps_shops: int
    total_beneficiaries: int
    transactions_this_month: int
    fraud_alerts_open: int
    fraud_alerts_critical: int
    avg_forecast_accuracy: float
    beneficiaries_within_3km_pct: float
    districts_covered: int
    top_fraud_districts: List[Dict[str, Any]]
    monthly_distribution_kg: Dict[str, float]
    last_updated: datetime
