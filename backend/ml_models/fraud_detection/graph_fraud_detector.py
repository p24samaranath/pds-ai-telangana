"""
Graph-Based Fraud Ring Detector for PDS Transactions

Uses NetworkX to build a bipartite transaction graph (beneficiary cards ↔ FPS shops)
and applies graph analytics to surface coordinated fraud rings — groups of cards and
shops that exhibit suspicious shared-transaction patterns.

Replaces the need for PyTorch Geometric (a GNN library) with an interpretable,
free alternative that surfaces the same fraud ring patterns using:
  1. Bipartite Graph Construction — cards and shops as nodes, transactions as edges
  2. Community Detection — Louvain-style iterative label propagation (NetworkX built-in)
  3. Ring Scoring — per-community anomaly score based on structural graph features
  4. PageRank Centrality — identifies the "hub" shop/card driving a fraud ring

No additional pip packages required — only networkx, numpy, pandas (already installed).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────────

# Minimum community size to be considered a suspicious ring
MIN_RING_SIZE = 3
# Minimum edge weight (transaction count between a card and shop) to add edge
MIN_EDGE_WEIGHT = 1
# Ring score threshold for High/Critical classification
RING_SCORE_HIGH = 0.65
RING_SCORE_CRITICAL = 0.80


# ── Main Detector ──────────────────────────────────────────────────────────────

class GraphFraudRingDetector:
    """
    Builds a bipartite card-shop transaction graph and detects fraud rings
    via community detection + structural anomaly scoring.

    Usage
    -----
    detector = GraphFraudRingDetector()
    rings, alerts = detector.detect_rings(transactions_df)
    """

    def __init__(
        self,
        min_ring_size: int = MIN_RING_SIZE,
        min_edge_weight: int = MIN_EDGE_WEIGHT,
        ring_score_critical: float = RING_SCORE_CRITICAL,
        ring_score_high: float = RING_SCORE_HIGH,
    ):
        self.min_ring_size = min_ring_size
        self.min_edge_weight = min_edge_weight
        self.ring_score_critical = ring_score_critical
        self.ring_score_high = ring_score_high

        # Populated after detect_rings()
        self.graph: Optional[nx.Graph] = None
        self.communities: List[set] = []
        self.ring_report: List[Dict[str, Any]] = []

    # ── Graph Construction ─────────────────────────────────────────────────────

    def _build_graph(self, df: pd.DataFrame) -> nx.Graph:
        """
        Build a weighted bipartite graph:
          - Node type A: beneficiary card IDs  (prefix 'c:')
          - Node type B: FPS shop IDs          (prefix 's:')
          - Edge weight: number of transactions between card and shop
          - Edge attributes: total_qty, unique_dates, biometric_miss_count
        """
        G = nx.Graph()

        required = {"card_id", "fps_shop_id", "quantity_kg"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"GraphFraudRingDetector: missing columns {missing}")

        df = df.copy()

        # Safe column initialisation — avoid KeyError on optional columns
        if "transaction_date" in df.columns:
            df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
        else:
            df["transaction_date"] = pd.NaT

        if "biometric_verified" in df.columns:
            df["biometric_verified"] = df["biometric_verified"].fillna(True).astype(bool)
        else:
            df["biometric_verified"] = True

        # Drop rows with null card_id or shop_id — they cannot form valid edges
        df = df.dropna(subset=["card_id", "fps_shop_id"])
        if df.empty:
            logger.warning("GraphFraudRingDetector._build_graph: all rows had null card_id/fps_shop_id")
            return G

        # Aggregate edge stats
        edge_stats: Dict[Tuple[str, str], Dict] = defaultdict(
            lambda: {"weight": 0, "total_qty": 0.0, "unique_dates": set(), "bio_miss": 0}
        )

        for _, row in df.iterrows():
            c_node = f"c:{row['card_id']}"
            s_node = f"s:{row['fps_shop_id']}"
            key = (c_node, s_node)
            stats = edge_stats[key]
            stats["weight"] += 1
            qty = row["quantity_kg"]
            stats["total_qty"] += float(qty) if pd.notna(qty) else 0.0
            if pd.notna(row["transaction_date"]):
                stats["unique_dates"].add(row["transaction_date"].date())
            if not bool(row["biometric_verified"]):
                stats["bio_miss"] += 1

        # Add nodes with type attribute for bipartite analysis
        for (c_node, s_node), stats in edge_stats.items():
            if stats["weight"] < self.min_edge_weight:
                continue

            if not G.has_node(c_node):
                G.add_node(c_node, node_type="card", raw_id=c_node[2:])
            if not G.has_node(s_node):
                G.add_node(s_node, node_type="shop", raw_id=s_node[2:])

            G.add_edge(
                c_node,
                s_node,
                weight=stats["weight"],
                total_qty=round(stats["total_qty"], 2),
                unique_dates=len(stats["unique_dates"]),
                bio_miss=stats["bio_miss"],
            )

        logger.info(
            f"GraphFraudRingDetector: built graph with {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges"
        )
        return G

    # ── Community Detection ────────────────────────────────────────────────────

    def _detect_communities(self, G: nx.Graph) -> List[set]:
        """
        Detect communities using NetworkX's built-in label propagation algorithm
        (greedy modularity or asynchronous label propagation — both free, no extra deps).

        Falls back to connected-component analysis if graph is too small.
        """
        if G.number_of_nodes() < 2:
            return []

        try:
            # Greedy modularity maximisation — best quality, O(n log^2 n)
            communities = list(nx.community.greedy_modularity_communities(G, weight="weight"))
        except Exception:
            try:
                # Async label propagation — faster, approximate
                communities = list(nx.community.asyn_lpa_communities(G, weight="weight"))
            except Exception:
                # Final fallback: connected components
                communities = list(nx.connected_components(G))

        # Filter communities that have at least min_ring_size nodes
        communities = [c for c in communities if len(c) >= self.min_ring_size]
        logger.info(f"GraphFraudRingDetector: found {len(communities)} communities (min size {self.min_ring_size})")
        return communities

    # ── Ring Scoring ───────────────────────────────────────────────────────────

    def _score_community(self, G: nx.Graph, community: set) -> Dict[str, Any]:
        """
        Compute an anomaly score for a detected community based on:
          1. multi_shop_cards    — cards transacting at > 1 shop (classic ring signal)
          2. bio_miss_rate       — fraction of edges with biometric failures
          3. density             — edge density vs expected bipartite density
          4. avg_weight          — mean transactions per card-shop pair
          5. pagerank_max        — highest PageRank node (hub shop/card)

        Returns a dict with score (0-1), component breakdown, and node lists.
        """
        subG = G.subgraph(community).copy()
        edges = list(subG.edges(data=True))
        nodes = list(subG.nodes(data=True))

        card_nodes = [n for n, d in nodes if d.get("node_type") == "card"]
        shop_nodes = [n for n, d in nodes if d.get("node_type") == "shop"]

        n_cards = len(card_nodes)
        n_shops = len(shop_nodes)

        if n_cards == 0 or n_shops == 0 or not edges:
            return {"score": 0.0}

        # --- Feature 1: multi-shop card ratio ---
        card_shop_count: Dict[str, int] = defaultdict(int)
        for u, v, _ in edges:
            if G.nodes[u].get("node_type") == "card":
                card_shop_count[u] += 1
            else:
                card_shop_count[v] += 1
        multi_shop_cards = sum(1 for cnt in card_shop_count.values() if cnt > 1)
        multi_shop_ratio = multi_shop_cards / n_cards  # 0-1

        # --- Feature 2: biometric miss rate ---
        total_weight = sum(d.get("weight", 1) for _, _, d in edges)
        total_bio_miss = sum(d.get("bio_miss", 0) for _, _, d in edges)
        bio_miss_rate = total_bio_miss / max(total_weight, 1)  # 0-1

        # --- Feature 3: density vs expected bipartite ---
        actual_edges = len(edges)
        max_edges = n_cards * n_shops
        density = actual_edges / max_edges  # 0-1; high density = tightly coupled ring

        # --- Feature 4: mean transaction weight per edge ---
        avg_weight = total_weight / actual_edges
        # Normalise: >10 transactions per pair is suspicious, cap at 1.0
        norm_avg_weight = min(avg_weight / 10.0, 1.0)

        # --- Feature 5: PageRank-based hub score ---
        try:
            pr = nx.pagerank(subG, weight="weight")
            pr_max = max(pr.values())
            # Centralised hub (>0.5 of weight going through one node) is suspicious
            hub_score = min(pr_max * n_cards, 1.0)  # normalise
        except Exception:
            hub_score = 0.0

        # --- Weighted composite score ---
        weights = {
            "multi_shop_ratio": 0.30,
            "bio_miss_rate": 0.25,
            "density": 0.20,
            "norm_avg_weight": 0.15,
            "hub_score": 0.10,
        }
        score = (
            weights["multi_shop_ratio"] * multi_shop_ratio
            + weights["bio_miss_rate"] * bio_miss_rate
            + weights["density"] * density
            + weights["norm_avg_weight"] * norm_avg_weight
            + weights["hub_score"] * hub_score
        )
        score = round(min(score, 1.0), 4)

        # --- PageRank-ranked hub node ---
        try:
            pr = nx.pagerank(subG, weight="weight")
            hub_node = max(pr, key=pr.get)
            hub_node_type = G.nodes[hub_node].get("node_type", "unknown")
            hub_node_id = G.nodes[hub_node].get("raw_id", hub_node)
        except Exception:
            hub_node_type, hub_node_id = "unknown", None

        return {
            "score": score,
            "n_cards": n_cards,
            "n_shops": n_shops,
            "n_edges": actual_edges,
            "total_transactions": int(total_weight),
            "multi_shop_ratio": round(multi_shop_ratio, 3),
            "bio_miss_rate": round(bio_miss_rate, 3),
            "density": round(density, 3),
            "avg_transactions_per_pair": round(avg_weight, 2),
            "hub_node_type": hub_node_type,
            "hub_node_id": hub_node_id,
            "card_ids": [G.nodes[n].get("raw_id", n) for n in card_nodes],
            "shop_ids": [G.nodes[n].get("raw_id", n) for n in shop_nodes],
        }

    # ── Alert Generation ───────────────────────────────────────────────────────

    def _community_to_alert(self, ring_id: int, score_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a scored community dict to a canonical alert dict."""
        score = score_data["score"]
        n_cards = score_data.get("n_cards", 0)
        n_shops = score_data.get("n_shops", 0)
        hub_type = score_data.get("hub_node_type", "unknown")
        hub_id = score_data.get("hub_node_id")
        bio_miss = score_data.get("bio_miss_rate", 0)

        # Severity
        if score >= self.ring_score_critical:
            severity = "Critical"
        elif score >= self.ring_score_high:
            severity = "High"
        else:
            severity = "Medium"

        # Human-readable explanation
        reasons = []
        if score_data.get("multi_shop_ratio", 0) > 0.3:
            reasons.append(
                f"{int(score_data['multi_shop_ratio'] * n_cards)} of {n_cards} cards "
                f"transacted at multiple shops (ghost beneficiary signal)"
            )
        if bio_miss > 0.2:
            reasons.append(
                f"{bio_miss:.0%} of transactions lacked biometric verification"
            )
        if score_data.get("density", 0) > 0.5:
            reasons.append(
                f"tightly coupled ring: {n_cards} cards × {n_shops} shops with "
                f"{score_data.get('n_edges', 0)} shared transactions"
            )
        if hub_id:
            reasons.append(
                f"hub {hub_type} '{hub_id}' has disproportionately high PageRank centrality"
            )
        if not reasons:
            reasons.append("statistically anomalous graph structure")

        explanation = (
            f"Graph fraud ring detected (ring_id={ring_id}, score={score:.2f}): "
            + "; ".join(reasons) + "."
        )

        shop_ids = score_data.get("shop_ids") or []
        card_ids = score_data.get("card_ids") or []

        return {
            "ring_id": ring_id,
            "anomaly_score": score,
            "severity": severity,
            "pattern": "coordinated_ring",
            "fraud_pattern": "coordinated_ring",
            "model": "graph_fraud_ring_detector",
            "fps_shop_id": shop_ids[0] if shop_ids else None,
            "card_id": None,  # Ring-level alert; no single card — use ring_metadata.card_ids
            "transaction_ids": [],  # Graph-level detection; no single transaction
            "explanation": explanation,
            "recommended_action": (
                "Immediately audit all flagged shops and freeze cards in ring. "
                "Cross-reference with ration card database for ghost beneficiaries."
            ),
            "ring_metadata": {
                "card_ids": score_data.get("card_ids", []),
                "shop_ids": score_data.get("shop_ids", []),
                "n_cards": n_cards,
                "n_shops": n_shops,
                "total_transactions": score_data.get("total_transactions", 0),
                "multi_shop_ratio": score_data.get("multi_shop_ratio", 0),
                "bio_miss_rate": score_data.get("bio_miss_rate", 0),
                "density": score_data.get("density", 0),
                "hub_node_type": hub_type,
                "hub_node_id": hub_id,
            },
        }

    # ── Public API ─────────────────────────────────────────────────────────────

    def detect_rings(
        self, transactions_df: pd.DataFrame
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Main entry point.

        Parameters
        ----------
        transactions_df : pd.DataFrame
            Must contain columns: card_id, fps_shop_id, quantity_kg
            Optional: transaction_date, biometric_verified

        Returns
        -------
        rings : list of ring summary dicts (one per detected ring)
        alerts : list of alert dicts in canonical FraudDetectionAgent format
        """
        if len(transactions_df) < self.min_ring_size:
            logger.info("GraphFraudRingDetector: insufficient data, skipping")
            return [], []

        try:
            G = self._build_graph(transactions_df)
            self.graph = G

            if G.number_of_edges() == 0:
                logger.info("GraphFraudRingDetector: no edges — graph is empty")
                return [], []

            communities = self._detect_communities(G)
            self.communities = communities

            rings = []
            alerts = []
            for ring_id, community in enumerate(communities):
                score_data = self._score_community(G, community)
                if score_data["score"] <= 0:
                    logger.debug(f"Ring {ring_id}: score=0, skipping (pure noise community)")
                    continue
                rings.append({"ring_id": ring_id, **score_data})
                alert = self._community_to_alert(ring_id, score_data)
                alerts.append(alert)

            # Sort by score descending
            rings.sort(key=lambda r: r["score"], reverse=True)
            alerts.sort(key=lambda a: a["anomaly_score"], reverse=True)

            self.ring_report = rings
            logger.info(
                f"GraphFraudRingDetector: detected {len(rings)} rings, "
                f"{len(alerts)} alerts generated"
            )
            return rings, alerts

        except Exception as exc:
            logger.error(f"GraphFraudRingDetector failed: {exc}", exc_info=True)
            return [], []

    def get_graph_summary(self) -> Dict[str, Any]:
        """Return high-level graph statistics after detect_rings() has been called."""
        if self.graph is None:
            return {"status": "not_run"}
        G = self.graph
        return {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "card_nodes": sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "card"),
            "shop_nodes": sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "shop"),
            "communities_detected": len(self.communities),
            "rings_flagged": len(self.ring_report),
            "density": round(nx.density(G), 4),
            "connected_components": nx.number_connected_components(G),
        }
