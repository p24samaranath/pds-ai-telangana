"""
Data Ingestion Service
======================
Priority order for each dataset:
  1. Real CSVs in data/raw/ produced by TelanganaFetcher (fps_shops.csv,
     transactions.csv, beneficiaries.csv)
  2. Any individual monthly CSVs (transactions_YYYY_MM.csv, etc.) left by
     a partial download — these are concatenated automatically
  3. Synthetic demo data (generated in-memory, no files needed)

Real column names from Open Data Telangana (verified 2025-07):
  fps_shops.csv     → shop_id, district, latitude, longitude, is_active, …
  transactions.csv  → shop_id, district, month, year, total_cards,
                      rice_afsc_kg, rice_fsc_kg, wheat_kg, sugar_kg,
                      red_gram_dal_kg, kerosene_litres, …
  beneficiaries.csv → shop_id, district, month, year, total_cards,
                      cards_nfsa_aay, cards_nfsa_phh, …
"""

import glob
import logging
import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from app.constants import CardType, CommodityType, TransactionStatus, TELANGANA_DISTRICTS

logger = logging.getLogger(__name__)

# ── District centre coordinates (lat, lon) ─────────────────────────────────────
DISTRICT_COORDS = {
    "Hyderabad":   (17.385, 78.487),
    "Nizamabad":   (18.672, 78.094),
    "Warangal":    (17.977, 79.596),
    "Khammam":     (17.247, 80.150),
    "Nalgonda":    (17.057, 79.267),
    "Karimnagar":  (18.438, 79.131),
    "Mahbubnagar": (16.737, 77.983),
    "Adilabad":    (19.664, 78.531),
    "Rangareddy":  (17.310, 78.420),
    "Medak":       (18.047, 78.265),
}

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _glob_concat(pattern: str) -> Optional[pd.DataFrame]:
    """Concatenate all CSVs matching a glob pattern. Returns None if none found."""
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f, on_bad_lines="skip", low_memory=False))
        except Exception as e:
            logger.warning(f"Skipping {f}: {e}")
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    logger.info(f"Loaded {len(files)} file(s) matching {pattern} → {len(df):,} rows")
    return df


def _normalise_shops(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure fps_shops has the columns expected downstream.
    Works whether the source is the real geo CSV or legacy synthetic data.
    """
    df = df.copy()

    # Real geo CSV uses 'is_active_str' already converted to bool by fetcher
    if "is_active" not in df.columns:
        if "is_active_str" in df.columns:
            df["is_active"] = df["is_active_str"].str.strip().str.lower() == "active"
        elif "fpsStatus" in df.columns:
            df["is_active"] = df["fpsStatus"].str.strip().str.lower() == "active"
        else:
            df["is_active"] = True

    # shop_id must be string
    if "shop_id" not in df.columns and "shopNo" in df.columns:
        df["shop_id"] = df["shopNo"].astype(str)
    df["shop_id"] = df["shop_id"].astype(str)

    # shop_name fallback
    if "shop_name" not in df.columns:
        df["shop_name"] = df.get("district", "Unknown") + " FPS " + df["shop_id"]

    # total_cards fallback (real geo file doesn't include it; will join from card_status)
    if "total_cards" not in df.columns:
        df["total_cards"] = 0

    return df


def _normalise_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the wide-format real transaction CSV into a long-format
    per-commodity DataFrame that the agents expect.

    Real columns: shop_id, district, month, year,
                  rice_afsc_kg, rice_fsc_kg, rice_aap_kg,
                  wheat_kg, sugar_kg, red_gram_dal_kg, kerosene_litres
    Output columns: fps_shop_id, district, commodity, quantity_kg,
                    transaction_date, card_id, status, biometric_verified
    """
    df = df.copy()

    # Ensure shop_id is string
    id_col = "shop_id" if "shop_id" in df.columns else ("shopNo" if "shopNo" in df.columns else None)
    if id_col:
        df["fps_shop_id"] = df[id_col].astype(str)
    else:
        df["fps_shop_id"] = "UNKNOWN"

    # Build transaction_date from month + year
    if "transaction_date" not in df.columns:
        month_col = "month" if "month" in df.columns else "source_month"
        year_col  = "year"  if "year"  in df.columns else "source_year"
        if month_col in df.columns and year_col in df.columns:
            df["transaction_date"] = pd.to_datetime(
                df[year_col].astype(str) + "-"
                + df[month_col].astype(str).str.zfill(2) + "-01",
                errors="coerce",
            )
        else:
            df["transaction_date"] = datetime.utcnow()

    # Map wide commodity columns → long rows
    commodity_map = {
        "rice_afsc_kg":       "rice",
        "rice_fsc_kg":        "rice",
        "rice_aap_kg":        "rice",
        "wheat_kg":           "wheat",
        "sugar_kg":           "sugar",
        "red_gram_dal_kg":    "red_gram_dal",
        "kerosene_litres":    "kerosene",
        "salt_kg":            "salt",
    }
    present = {col: comm for col, comm in commodity_map.items() if col in df.columns}

    if present:
        keep_cols = ["fps_shop_id", "district", "transaction_date", "total_cards"]
        keep_cols = [c for c in keep_cols if c in df.columns]
        rows = []
        for col, commodity in present.items():
            sub = df[keep_cols + [col]].copy()
            sub = sub[sub[col].notna() & (sub[col] > 0)]
            sub["commodity"]     = commodity
            sub["quantity_kg"]   = sub[col]
            sub["card_id"]       = sub["fps_shop_id"] + "_card"
            sub["status"]        = TransactionStatus.COMPLETED.value
            sub["biometric_verified"] = True
            sub["transaction_id"] = (
                sub["fps_shop_id"] + "_" + commodity + "_"
                + sub["transaction_date"].astype(str)
            )
            rows.append(sub.drop(columns=[col]))
        if rows:
            df = pd.concat(rows, ignore_index=True)
    else:
        # Already long format — just ensure required columns exist
        if "commodity"   not in df.columns:
            df["commodity"] = "rice"
        if "quantity_kg" not in df.columns:
            df["quantity_kg"] = 0.0
        if "card_id"     not in df.columns:
            df["card_id"] = df["fps_shop_id"] + "_card"
        if "status"      not in df.columns:
            df["status"] = TransactionStatus.COMPLETED.value
        if "biometric_verified" not in df.columns:
            df["biometric_verified"] = True
        if "transaction_id" not in df.columns:
            df["transaction_id"] = df.get("fps_shop_id", "TXN") + "_" + df.index.astype(str)

    return df


def _normalise_beneficiaries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the card_status CSV into a beneficiary-like DataFrame.
    Real columns: shop_id, district, month, year, total_cards,
                  cards_nfsa_aay, cards_nfsa_phh, …
    """
    df = df.copy()

    id_col = "shop_id" if "shop_id" in df.columns else "shopNo"
    if id_col in df.columns:
        df["fps_shop_id"] = df[id_col].astype(str)

    if "district" not in df.columns and "distName" in df.columns:
        df["district"] = df["distName"]

    # Derive a synthetic card_id if not present
    if "card_id" not in df.columns:
        df["card_id"] = df.get("fps_shop_id", "shop") + "_" + df.index.astype(str)

    # card_type — derive from NFSA AAY count
    if "card_type" not in df.columns:
        aay_col = "cards_nfsa_aay" if "cards_nfsa_aay" in df.columns else None
        df["card_type"] = "PHH"
        if aay_col:
            df.loc[df[aay_col] > 0, "card_type"] = "AAY"

    if "members_count" not in df.columns:
        units_col = "total_units" if "total_units" in df.columns else None
        cards_col = "total_cards" if "total_cards" in df.columns else None
        if units_col and cards_col:
            df["members_count"] = (
                df[units_col] / df[cards_col].replace(0, 1)
            ).round().astype(int).clip(1, 10)
        else:
            df["members_count"] = 3

    if "is_active" not in df.columns:
        df["is_active"] = True

    if "latitude" not in df.columns:
        df["latitude"] = None
    if "longitude" not in df.columns:
        df["longitude"] = None

    return df


# ── Main service ───────────────────────────────────────────────────────────────

class DataIngestionService:
    """
    Loads PDS data in this priority order:
      1. Master CSVs written by TelanganaFetcher (fps_shops.csv, transactions.csv, beneficiaries.csv)
      2. Individual monthly CSVs from partial downloads (transactions_YYYY_MM.csv, etc.)
      3. Synthetic demo data (no files needed)
    """

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else RAW_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ── Public loaders ────────────────────────────────────────────────────────

    def load_shops(self) -> pd.DataFrame:
        # 1. Master file
        master = self.data_dir / "fps_shops.csv"
        if master.exists():
            logger.info(f"Loading shops from {master}")
            return _normalise_shops(pd.read_csv(master, low_memory=False))

        # 2. Monthly geo files
        df = _glob_concat(str(self.data_dir / "geo_locations_*.csv"))
        if df is not None:
            return _normalise_shops(df)

        # 3. Synthetic
        logger.info("No shops data found — using synthetic data")
        return self._synthetic_shops()

    def load_transactions(self) -> pd.DataFrame:
        # 1. Master file
        master = self.data_dir / "transactions.csv"
        if master.exists():
            logger.info(f"Loading transactions from {master}")
            df = pd.read_csv(master, low_memory=False)
            return _normalise_transactions(df)

        # 2. Monthly files
        df = _glob_concat(str(self.data_dir / "transactions_*.csv"))
        if df is not None:
            return _normalise_transactions(df)

        # 3. Synthetic
        logger.info("No transaction data found — using synthetic data")
        return self._synthetic_transactions()

    def load_beneficiaries(self) -> pd.DataFrame:
        # 1. Master file
        master = self.data_dir / "beneficiaries.csv"
        if master.exists():
            logger.info(f"Loading beneficiaries from {master}")
            df = pd.read_csv(master, low_memory=False)
            return _normalise_beneficiaries(df)

        # 2. Monthly card_status files
        df = _glob_concat(str(self.data_dir / "card_status_*.csv"))
        if df is not None:
            return _normalise_beneficiaries(df)

        # 3. Synthetic
        logger.info("No beneficiary data found — using synthetic data")
        return self._synthetic_beneficiaries()

    def get_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        shops         = self.load_shops()
        transactions  = self.load_transactions()
        beneficiaries = self.load_beneficiaries()
        return shops, beneficiaries, transactions

    def data_status(self) -> dict:
        """Return what data is currently loaded and its shape / date range."""
        shops        = self.load_shops()
        txn          = self.load_transactions()
        bene         = self.load_beneficiaries()

        def _date_range(df: pd.DataFrame) -> str:
            if "transaction_date" in df.columns:
                s = pd.to_datetime(df["transaction_date"], errors="coerce").dropna()
                if len(s):
                    return f"{s.min().date()} → {s.max().date()}"
            if "source_year" in df.columns and "source_month" in df.columns:
                lo = df[["source_year","source_month"]].dropna()
                if len(lo):
                    mi = lo.min()
                    ma = lo.max()
                    return f"{int(mi.source_year)}-{int(mi.source_month):02d} → {int(ma.source_year)}-{int(ma.source_month):02d}"
            return "N/A"

        real_shops = (self.data_dir / "fps_shops.csv").exists()
        real_txn   = (self.data_dir / "transactions.csv").exists()
        real_bene  = (self.data_dir / "beneficiaries.csv").exists()

        return {
            "fps_shops":      {"rows": len(shops),  "source": "real" if real_shops else "synthetic"},
            "transactions":   {"rows": len(txn),    "source": "real" if real_txn   else "synthetic",
                               "date_range": _date_range(txn)},
            "beneficiaries":  {"rows": len(bene),   "source": "real" if real_bene  else "synthetic"},
            "data_dir":       str(self.data_dir),
        }

    # ── Synthetic generators (unchanged, used as fallback) ────────────────────

    def _synthetic_shops(self, n: int = 150) -> pd.DataFrame:
        random.seed(42); np.random.seed(42)
        rows = []
        districts = list(DISTRICT_COORDS.keys())
        for i in range(n):
            d = random.choice(districts)
            blat, blon = DISTRICT_COORDS[d]
            cards = random.randint(100, 800)
            rows.append({
                "shop_id": f"TG{d[:3].upper()}{i+1:04d}",
                "shop_name": f"{d} FPS Shop {i+1}",
                "district": d,
                "mandal": f"Mandal {random.randint(1,10)}",
                "latitude":  round(blat + np.random.uniform(-0.5, 0.5), 6),
                "longitude": round(blon + np.random.uniform(-0.5, 0.5), 6),
                "is_active": random.random() > 0.05,
                "total_cards": cards,
            })
        return pd.DataFrame(rows)

    def _synthetic_beneficiaries(self, n: int = 5000) -> pd.DataFrame:
        random.seed(42); np.random.seed(42)
        districts = list(DISTRICT_COORDS.keys())
        rows = []
        for i in range(n):
            d = random.choice(districts)
            blat, blon = DISTRICT_COORDS[d]
            rows.append({
                "card_id":      f"TG{i+1:07d}",
                "card_type":    random.choice([CardType.AAY.value, CardType.PHH.value, CardType.PHH.value]),
                "fps_shop_id":  f"TG{d[:3].upper()}{random.randint(1,15):04d}",
                "district":     d,
                "members_count": random.randint(1, 6),
                "is_active":    random.random() > 0.03,
                "latitude":  round(blat + np.random.uniform(-0.6, 0.6), 6),
                "longitude": round(blon + np.random.uniform(-0.6, 0.6), 6),
            })
        return pd.DataFrame(rows)

    def _synthetic_transactions(self, n_months: int = 12) -> pd.DataFrame:
        random.seed(42); np.random.seed(42)
        shops = self._synthetic_shops(50)
        bene  = self._synthetic_beneficiaries(2000)
        commodities = [CommodityType.RICE.value, CommodityType.WHEAT.value]
        rows, txn_id = [], 1
        base = datetime.now() - timedelta(days=n_months * 30)
        for mo in range(n_months):
            mstart = base + timedelta(days=mo * 30)
            sample = bene.sample(min(1500, len(bene)), random_state=mo)
            for _, b in sample.iterrows():
                for comm in commodities:
                    qty  = max(1.0, round(np.random.normal(35 if comm=="rice" else 10, 5), 1))
                    hour = random.randint(8, 19)
                    if random.random() < 0.05:                       # inject fraud
                        fraud = random.choice(["after_hours","bulk","high_vol"])
                        if fraud == "after_hours": hour = random.choice([0,1,2,23])
                        elif fraud == "high_vol":  qty *= 3
                    day = random.randint(1, 28)
                    rows.append({
                        "transaction_id":    f"TXN{txn_id:010d}",
                        "card_id":           b["card_id"],
                        "fps_shop_id":       b["fps_shop_id"],
                        "commodity":         comm,
                        "quantity_kg":       qty,
                        "transaction_date":  mstart + timedelta(days=day-1, hours=hour),
                        "status":            TransactionStatus.COMPLETED.value,
                        "biometric_verified": random.random() > 0.15,
                        "epos_device_id":    f"EPOS{random.randint(1,500):05d}",
                    })
                    txn_id += 1
        df = pd.DataFrame(rows)
        logger.info(f"Synthetic: {len(df):,} transactions over {n_months} months")
        return df
