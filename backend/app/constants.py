from enum import Enum


class CardType(str, Enum):
    AAY = "AAY"           # Antyodaya Anna Yojana (poorest of poor)
    PHH = "PHH"           # Priority Household
    NPHH = "NPHH"         # Non-Priority Household
    ANNAPURNA = "Annapurna"


class CommodityType(str, Enum):
    RICE = "rice"
    WHEAT = "wheat"
    KEROSENE = "kerosene"
    SUGAR = "sugar"
    RED_GRAM_DAL = "red_gram_dal"


class FraudSeverity(str, Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class FraudPattern(str, Enum):
    GHOST_BENEFICIARY = "ghost_beneficiary"
    DUPLICATE_TRANSACTION = "duplicate_transaction"
    DEALER_SKIMMING = "dealer_skimming"
    TEMPORAL_ANOMALY = "temporal_anomaly"
    COORDINATED_RING = "coordinated_ring"


class TransactionStatus(str, Enum):
    COMPLETED = "completed"
    PENDING = "pending"
    FLAGGED = "flagged"
    BLOCKED = "blocked"
    REVERSED = "reversed"


class AlertStatus(str, Enum):
    OPEN = "open"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"
    ESCALATED = "escalated"


class AgentName(str, Enum):
    ORCHESTRATOR = "orchestrator"
    DEMAND_FORECAST = "demand_forecast"
    FRAUD_DETECTION = "fraud_detection"
    GEOSPATIAL = "geospatial"
    REPORTING = "reporting"


# Commodity monthly allocations (kg per family)
COMMODITY_ALLOCATIONS = {
    CardType.AAY: {
        CommodityType.RICE: 35.0,
        CommodityType.WHEAT: 0.0,
        CommodityType.SUGAR: 1.0,
        CommodityType.KEROSENE: 3.0,
    },
    CardType.PHH: {
        CommodityType.RICE: 5.0,   # per person
        CommodityType.WHEAT: 0.0,
        CommodityType.SUGAR: 0.5,
        CommodityType.KEROSENE: 3.0,
    },
}

TELANGANA_DISTRICTS = [
    "Hyderabad", "Rangareddy", "Medchal-Malkajgiri", "Vikarabad",
    "Sangareddy", "Medak", "Nizamabad", "Kamareddy", "Nirmal",
    "Adilabad", "Mancherial", "Kumuram Bheem", "Jagityal",
    "Karimnagar", "Rajanna Sircilla", "Peddapalli", "Jayashankar",
    "Mulugu", "Bhadradri Kothagudem", "Khammam", "Mahabubabad",
    "Warangal", "Hanamkonda", "Jangaon", "Yadadri Bhuvanagiri",
    "Nalgonda", "Suryapet", "Mahabubnagar", "Nagarkurnool",
    "Wanaparthy", "Gadwal", "Narayanpet", "Siddipet",
]
