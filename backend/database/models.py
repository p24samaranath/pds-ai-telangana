from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Enum as SAEnum,
    ForeignKey, Text, JSON
)
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime
from app.constants import CardType, CommodityType, FraudSeverity, TransactionStatus, AlertStatus

Base = declarative_base()


class FPSShop(Base):
    __tablename__ = "fps_shops"

    id = Column(Integer, primary_key=True, index=True)
    shop_id = Column(String(50), unique=True, index=True, nullable=False)
    shop_name = Column(String(200), nullable=False)
    dealer_name = Column(String(200))
    dealer_id = Column(String(50), index=True)
    district = Column(String(100), index=True)
    mandal = Column(String(100))
    village = Column(String(200))
    address = Column(Text)
    latitude = Column(Float)
    longitude = Column(Float)
    is_active = Column(Boolean, default=True)
    activation_date = Column(DateTime)
    deactivation_date = Column(DateTime)
    total_cards = Column(Integer, default=0)
    monthly_allocation_rice_kg = Column(Float, default=0.0)
    monthly_allocation_wheat_kg = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    transactions = relationship("Transaction", back_populates="fps_shop")
    beneficiaries = relationship("Beneficiary", back_populates="fps_shop")
    fraud_alerts = relationship("FraudAlert", back_populates="fps_shop")
    forecasts = relationship("DemandForecast", back_populates="fps_shop")


class Beneficiary(Base):
    __tablename__ = "beneficiaries"

    id = Column(Integer, primary_key=True, index=True)
    card_id = Column(String(50), unique=True, index=True, nullable=False)
    aadhaar_token = Column(String(100), unique=True)  # Tokenized, not raw
    head_of_family = Column(String(200))
    card_type = Column(SAEnum(CardType), nullable=False)
    fps_shop_id = Column(Integer, ForeignKey("fps_shops.id"))
    district = Column(String(100), index=True)
    mandal = Column(String(100))
    village = Column(String(200))
    members_count = Column(Integer, default=1)
    is_active = Column(Boolean, default=True)
    mobile_number = Column(String(15))
    latitude = Column(Float)
    longitude = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    fps_shop = relationship("FPSShop", back_populates="beneficiaries")
    transactions = relationship("Transaction", back_populates="beneficiary")


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String(100), unique=True, index=True, nullable=False)
    beneficiary_id = Column(Integer, ForeignKey("beneficiaries.id"))
    fps_shop_id = Column(Integer, ForeignKey("fps_shops.id"))
    commodity = Column(SAEnum(CommodityType), nullable=False)
    quantity_kg = Column(Float, nullable=False)
    transaction_date = Column(DateTime, nullable=False, index=True)
    status = Column(SAEnum(TransactionStatus), default=TransactionStatus.COMPLETED)
    biometric_verified = Column(Boolean, default=False)
    epos_device_id = Column(String(50))
    anomaly_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    beneficiary = relationship("Beneficiary", back_populates="transactions")
    fps_shop = relationship("FPSShop", back_populates="transactions")
    fraud_alert = relationship("FraudAlert", back_populates="transaction", uselist=False)


class FraudAlert(Base):
    __tablename__ = "fraud_alerts"

    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(String(100), unique=True, index=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id"), nullable=True)
    fps_shop_id = Column(Integer, ForeignKey("fps_shops.id"), nullable=True)
    beneficiary_card_id = Column(String(50), index=True)
    severity = Column(SAEnum(FraudSeverity), nullable=False)
    fraud_pattern = Column(String(100))
    anomaly_score = Column(Float)
    description = Column(Text)
    explanation = Column(Text)       # Human-readable SHAP explanation
    recommended_action = Column(String(200))
    status = Column(SAEnum(AlertStatus), default=AlertStatus.OPEN)
    detected_at = Column(DateTime, default=datetime.utcnow, index=True)
    resolved_at = Column(DateTime, nullable=True)
    resolved_by = Column(String(100), nullable=True)
    metadata = Column(JSON, default={})

    transaction = relationship("Transaction", back_populates="fraud_alert")
    fps_shop = relationship("FPSShop", back_populates="fraud_alerts")


class DemandForecast(Base):
    __tablename__ = "demand_forecasts"

    id = Column(Integer, primary_key=True, index=True)
    fps_shop_id = Column(Integer, ForeignKey("fps_shops.id"))
    forecast_month = Column(DateTime, nullable=False, index=True)
    commodity = Column(SAEnum(CommodityType), nullable=False)
    predicted_quantity_kg = Column(Float, nullable=False)
    confidence_lower = Column(Float)
    confidence_upper = Column(Float)
    model_used = Column(String(50))    # lstm, prophet, ensemble
    mape_score = Column(Float, nullable=True)
    risk_flag = Column(String(50), nullable=True)  # overstock, understock
    generated_at = Column(DateTime, default=datetime.utcnow)

    fps_shop = relationship("FPSShop", back_populates="forecasts")


class GeospatialAnalysis(Base):
    __tablename__ = "geospatial_analyses"

    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(String(100), unique=True)
    district = Column(String(100), index=True)
    analysis_type = Column(String(50))  # underserved_zones, new_location, closure
    description = Column(Text)
    recommended_lat = Column(Float, nullable=True)
    recommended_lon = Column(Float, nullable=True)
    projected_coverage = Column(Integer, nullable=True)
    current_avg_distance_km = Column(Float, nullable=True)
    projected_avg_distance_km = Column(Float, nullable=True)
    priority_score = Column(Float, default=0.0)
    analysis_data = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)


class AgentLog(Base):
    __tablename__ = "agent_logs"

    id = Column(Integer, primary_key=True, index=True)
    agent_name = Column(String(100), nullable=False, index=True)
    action = Column(String(200), nullable=False)
    input_summary = Column(Text)
    output_summary = Column(Text)
    duration_ms = Column(Integer)
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)
    metadata = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
