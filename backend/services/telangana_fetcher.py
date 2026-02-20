"""
Open Data Telangana — Dataset Fetcher
======================================
Discovers every CSV file published across the three PDS datasets,
downloads them, normalises column names to match the project schema,
and saves clean CSVs to backend/data/raw/.

Real column mapping (verified from live API 2025-07):

TRANSACTIONS CSV (shop-wise-trans-details_M_YYYY.csv)
  distCode, distName, officeCode, officeName, shopNo,
  month, year, noOfRcs, noOfTrans,
  riceAfsc, riceFsc, riceAap, wheat, sugar, rgdal, kerosene,
  totalAmount, salt, otherShopTransCnt

CARD STATUS CSV (fpshop-card-status_M_YYYY.csv)
  distCode, distName, officeCode, officeName, shopNo,
  month, year,
  rcNfsaAay, unitsNfsaAay, rcNfsaPhh, unitsNfsaPhh,
  totalRcNfsa, totalUnitsNfsa,
  rcStateAay, unitsStateAay, rcStatePhh, unitsStatePhh,
  rcStateAap, unitsStateAap, totalRcState, totalUnitsState,
  totalRcs, totalUnits, cardTypeId, units, gasCylinders,
  mroAsoApprDate, cardPoolType

GEO LOCATIONS CSV (shop-status-details_M_YYYY.csv)
  distCode, distName, officeCode, officeName, shopNo,
  address, longitude, latitude, fpsStatus, fpsType, dateTime
"""

import io
import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

BASE_API  = "https://data.telangana.gov.in/api/1/metastore/schemas/dataset/items"
DATA_DIR  = Path(__file__).resolve().parents[1] / "data" / "raw"
TIMEOUT   = 30       # seconds per HTTP request
RATE_DELAY = 0.5     # seconds between requests (be a polite citizen)

DATASET_IDS: Dict[str, str] = {
    "transactions":  "4a7337ce-08bd-4f74-8e00-bd6eb83eb4ff",
    "card_status":   "84e47ac8-2c24-43a4-a045-2a7bec7212e5",
    "geo_locations": "6b5a1abe-6bf8-4c28-a400-2e42640de641",
}

# ── Column normalisers ─────────────────────────────────────────────────────────
# Map raw API column names → project-internal names used by data_ingestion.py

TRANSACTIONS_RENAME: Dict[str, str] = {
    "distCode":           "district_code",
    "distName":           "district",
    "officeCode":         "office_code",
    "officeName":         "office_name",
    "shopNo":             "shop_id",
    "month":              "month",
    "year":               "year",
    "noOfRcs":            "total_cards",
    "noOfTrans":          "total_transactions",
    "riceAfsc":           "rice_afsc_kg",
    "riceFsc":            "rice_fsc_kg",
    "riceAap":            "rice_aap_kg",
    "wheat":              "wheat_kg",
    "sugar":              "sugar_kg",
    "rgdal":              "red_gram_dal_kg",
    "kerosene":           "kerosene_litres",
    "totalAmount":        "total_amount",
    "salt":               "salt_kg",
    "otherShopTransCnt":  "other_shop_transactions",
}

CARD_STATUS_RENAME: Dict[str, str] = {
    "distCode":           "district_code",
    "distName":           "district",
    "officeCode":         "office_code",
    "officeName":         "office_name",
    "shopNo":             "shop_id",
    "month":              "month",
    "year":               "year",
    "rcNfsaAay":          "cards_nfsa_aay",
    "unitsNfsaAay":       "units_nfsa_aay",
    "rcNfsaPhh":          "cards_nfsa_phh",
    "unitsNfsaPhh":       "units_nfsa_phh",
    "totalRcNfsa":        "total_cards_nfsa",
    "totalUnitsNfsa":     "total_units_nfsa",
    "rcStateAay":         "cards_state_aay",
    "unitsStateAay":      "units_state_aay",
    "rcStatePhh":         "cards_state_phh",
    "unitsStatePhh":      "units_state_phh",
    "rcStateAap":         "cards_state_aap",
    "unitsStateAap":      "units_state_aap",
    "totalRcState":       "total_cards_state",
    "totalUnitsState":    "total_units_state",
    "totalRcs":           "total_cards",
    "totalUnits":         "total_units",
    "cardTypeId":         "card_type_id",
    "units":              "units",
    "gasCylinders":       "gas_cylinders",
    "mroAsoApprDate":     "approval_date",
    "cardPoolType":       "card_pool_type",
}

GEO_RENAME: Dict[str, str] = {
    "distCode":   "district_code",
    "distName":   "district",
    "officeCode": "office_code",
    "officeName": "office_name",
    "shopNo":     "shop_id",
    "address":    "address",
    "longitude":  "longitude",
    "latitude":   "latitude",
    "fpsStatus":  "is_active_str",
    "fpsType":    "shop_type",
    "dateTime":   "last_updated",
}


# ── Fetcher class ──────────────────────────────────────────────────────────────

class TelanganaFetcher:
    """
    Discovers and downloads PDS datasets from Open Data Telangana.

    Usage:
        fetcher = TelanganaFetcher()
        manifest = fetcher.discover()          # list all available files
        fetcher.download_latest()              # grab the most recent month only
        fetcher.download_range(2024, 1, 2025, 5)  # grab a date range
        fetcher.download_all()                 # grab everything (slow!)
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "PDS-AI-Optimizer/1.0 (research)"

    # ── Manifest discovery ────────────────────────────────────────────────────

    def discover(self) -> Dict[str, List[Dict]]:
        """
        Hit the metastore API for all three datasets and return a manifest
        of every downloadable file with its metadata.

        Returns:
            {
              "transactions": [{"title": ..., "url": ..., "month": 5, "year": 2025}, ...],
              "card_status":  [...],
              "geo_locations":[...],
            }
        """
        manifest: Dict[str, List[Dict]] = {}
        for name, uid in DATASET_IDS.items():
            url = f"{BASE_API}/{uid}?show-reference-ids"
            logger.info(f"Discovering {name} from {url}")
            try:
                resp = self._session.get(url, timeout=TIMEOUT)
                resp.raise_for_status()
                data = resp.json()
                files = []
                for dist in data.get("distribution", []):
                    d = dist.get("data", dist)
                    title = d.get("title", "")
                    download_url = d.get("downloadURL", "")
                    if not download_url:
                        continue
                    month, year = _extract_month_year(title, download_url)
                    files.append({
                        "title": title,
                        "url":   download_url,
                        "month": month,
                        "year":  year,
                    })
                manifest[name] = files
                logger.info(f"  Found {len(files)} files for {name}")
                time.sleep(RATE_DELAY)
            except Exception as e:
                logger.error(f"Discovery failed for {name}: {e}")
                manifest[name] = []
        return manifest

    # ── Download helpers ──────────────────────────────────────────────────────

    def _download_file(self, name: str, entry: Dict) -> Optional[pd.DataFrame]:
        """Download one CSV entry and return a normalised DataFrame, or None on error."""
        url   = entry["url"]
        month = entry.get("month")
        year  = entry.get("year")
        fname = f"{name}_{year}_{month:02d}.csv" if month and year else f"{name}_unknown.csv"
        dest  = self.data_dir / fname

        if dest.exists():
            logger.debug(f"  Already cached: {dest.name}")
            return pd.read_csv(dest)

        try:
            logger.info(f"  Downloading {dest.name} from {url}")
            r = self._session.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text), on_bad_lines="skip")
            df = _normalise(df, name)
            df["source_month"]  = month
            df["source_year"]   = year
            df["downloaded_at"] = datetime.utcnow().isoformat()
            df.to_csv(dest, index=False)
            time.sleep(RATE_DELAY)
            return df
        except Exception as e:
            logger.error(f"  Failed {url}: {e}")
            return None

    # ── Public download methods ───────────────────────────────────────────────

    def download_latest(self) -> Dict[str, pd.DataFrame]:
        """Download only the most recent file for each dataset."""
        manifest = self.discover()
        results: Dict[str, pd.DataFrame] = {}
        for name, files in manifest.items():
            if not files:
                continue
            # Sort by (year, month) descending and take the first
            valid = [f for f in files if f["year"] and f["month"]]
            if not valid:
                valid = files
            latest = sorted(valid, key=lambda x: (x["year"] or 0, x["month"] or 0), reverse=True)[0]
            df = self._download_file(name, latest)
            if df is not None:
                results[name] = df
                logger.info(f"Latest {name}: {len(df):,} rows — {latest['title']}")
        return results

    def download_range(
        self,
        from_year: int, from_month: int,
        to_year:   int, to_month:   int,
    ) -> Dict[str, pd.DataFrame]:
        """
        Download files whose (year, month) falls within the given range
        and concatenate them per dataset.

        Example:
            fetcher.download_range(2024, 1, 2024, 12)  # full year 2024
        """
        manifest = self.discover()
        results: Dict[str, pd.DataFrame] = {}
        from_key = from_year * 100 + from_month
        to_key   = to_year   * 100 + to_month

        for name, files in manifest.items():
            dfs = []
            for entry in files:
                if entry["year"] is None or entry["month"] is None:
                    continue
                key = entry["year"] * 100 + entry["month"]
                if from_key <= key <= to_key:
                    df = self._download_file(name, entry)
                    if df is not None:
                        dfs.append(df)
            if dfs:
                combined = pd.concat(dfs, ignore_index=True)
                results[name] = combined
                logger.info(f"{name}: {len(dfs)} files → {len(combined):,} rows")
        return results

    def download_all(self) -> Dict[str, pd.DataFrame]:
        """Download every file in the manifest (2018–present). Can be slow."""
        manifest = self.discover()
        results: Dict[str, pd.DataFrame] = {}
        for name, files in manifest.items():
            dfs = []
            for entry in files:
                df = self._download_file(name, entry)
                if df is not None:
                    dfs.append(df)
            if dfs:
                combined = pd.concat(dfs, ignore_index=True)
                results[name] = combined
                logger.info(f"{name}: {len(dfs)} files → {len(combined):,} rows total")
        return results

    # ── Build the merged master files ─────────────────────────────────────────

    def build_master_files(self, dataframes: Dict[str, pd.DataFrame]):
        """
        Save/append to the three master files read by data_ingestion.py:
            data/raw/fps_shops.csv
            data/raw/transactions.csv
            data/raw/beneficiaries.csv   (derived from card_status)
        """
        # 1. GEO → fps_shops.csv
        if "geo_locations" in dataframes:
            geo = dataframes["geo_locations"].copy()
            geo["is_active"] = geo["is_active_str"].str.strip().str.lower() == "active"
            geo = geo.drop(columns=["is_active_str"], errors="ignore")
            shops_path = self.data_dir / "fps_shops.csv"
            geo.to_csv(shops_path, index=False)
            logger.info(f"Saved fps_shops.csv  ({len(geo):,} rows)")

        # 2. TRANSACTIONS → transactions.csv
        if "transactions" in dataframes:
            txn = dataframes["transactions"].copy()
            txn["transaction_date"] = pd.to_datetime(
                txn["year"].astype(str) + "-" + txn["month"].astype(str).str.zfill(2) + "-01"
            )
            # Add a commodity column: longest column is usually rice_fsc_kg
            txn_path = self.data_dir / "transactions.csv"
            txn.to_csv(txn_path, index=False)
            logger.info(f"Saved transactions.csv ({len(txn):,} rows)")

        # 3. CARD STATUS → beneficiaries.csv (one row per shop per month)
        if "card_status" in dataframes:
            cs = dataframes["card_status"].copy()
            bene_path = self.data_dir / "beneficiaries.csv"
            cs.to_csv(bene_path, index=False)
            logger.info(f"Saved beneficiaries.csv ({len(cs):,} rows)")

    def fetch_and_save(
        self,
        mode: str = "latest",
        from_year: int = 2024, from_month: int = 1,
        to_year:   int = 2025, to_month:   int = 12,
    ) -> Dict[str, int]:
        """
        One-call convenience method.

        Args:
            mode: "latest" | "range" | "all"
            from_year, from_month, to_year, to_month — only used when mode="range"

        Returns:
            dict of {dataset_name: row_count}
        """
        logger.info(f"Starting fetch — mode={mode}")
        if mode == "latest":
            dfs = self.download_latest()
        elif mode == "range":
            dfs = self.download_range(from_year, from_month, to_year, to_month)
        else:
            dfs = self.download_all()

        self.build_master_files(dfs)
        summary = {k: len(v) for k, v in dfs.items()}
        logger.info(f"Fetch complete: {summary}")
        return summary


# ── Helpers ────────────────────────────────────────────────────────────────────

def _normalise(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Rename columns from raw API names to project-internal names."""
    rename_map = {
        "transactions":  TRANSACTIONS_RENAME,
        "card_status":   CARD_STATUS_RENAME,
        "geo_locations": GEO_RENAME,
    }.get(dataset_name, {})
    df = df.rename(columns=rename_map)
    return df


def _extract_month_year(title: str, url: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Pull (month, year) out of a file title or URL.
    Titles look like: "... Data 01-05-2025 to 31-05-2025"
    URLs look like:   ".../shop-wise-trans-details_5_2025.csv"
    """
    import re
    # Try URL pattern first: _M_YYYY.csv or _MM_YYYY.csv
    m = re.search(r"_(\d{1,2})_(\d{4})\.csv", url)
    if m:
        return int(m.group(1)), int(m.group(2))
    # Try title: "to DD-MM-YYYY"
    m = re.search(r"to\s+\d{1,2}-(\d{1,2})-(\d{4})", title)
    if m:
        return int(m.group(1)), int(m.group(2))
    # Try title: "01-MM-YYYY to ..."
    m = re.search(r"(\d{1,2})-(\d{4})", title)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None
