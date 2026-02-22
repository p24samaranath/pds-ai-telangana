"""
In-Memory RAG (Retrieval-Augmented Generation) Vector Store for PDS

Provides a lightweight, zero-dependency (no Pinecone/Weaviate/Chroma) retrieval
layer using TF-IDF vectors + cosine similarity. Documents are agent outputs
serialised as plain-English text chunks.

Design
------
- RAGStore.index()     — converts agent outputs → text chunks → TF-IDF matrix
- RAGStore.retrieve()  — query → top-k relevant chunks
- RAGStore.reset()     — clear store (call before each monthly pipeline run)

All storage is in-process memory (survives the lifetime of the FastAPI worker).
For production persistence, swap TF-IDF for a real vector DB — the interface is
identical.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class RAGDocument:
    """A single retrievable text chunk with its source metadata."""
    doc_id: str
    text: str
    source: str          # e.g. "fraud_alerts", "forecast_risk", "geo_analysis"
    metadata: Dict[str, Any] = field(default_factory=dict)
    indexed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ── Vector Store ───────────────────────────────────────────────────────────────

class RAGStore:
    """
    TF-IDF in-memory vector store.

    Usage
    -----
    store = RAGStore()
    store.index(fraud_results, forecast_results, geo_results, shops_df, bene_df)
    chunks = store.retrieve("Which district has the most fraud?", k=5)
    """

    def __init__(self, max_features: int = 5000, min_df: int = 1):
        self._docs: List[RAGDocument] = []
        self._matrix: Optional[np.ndarray] = None          # TF-IDF matrix (n_docs × n_features)
        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            stop_words="english",
            ngram_range=(1, 2),   # unigrams + bigrams for better matching
        )
        self._fitted = False

    # ── Indexing ───────────────────────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        """True if the index has been built and is ready for retrieval."""
        return self._fitted

    def reset(self) -> None:
        """Clear all indexed documents — call before re-indexing after a fresh pipeline run."""
        self._docs = []
        self._matrix = None
        self._fitted = False
        logger.info("RAGStore: cleared")

    def add_document(self, doc: RAGDocument) -> None:
        """Add a single document (invalidates the current index — call build_index() after)."""
        self._docs.append(doc)
        self._fitted = False

    def build_index(self) -> None:
        """(Re)build the TF-IDF matrix from all current documents."""
        if not self._docs:
            logger.warning("RAGStore.build_index: no documents to index")
            return
        texts = [d.text for d in self._docs]
        # Guard against all-empty texts which would make TF-IDF fail
        non_empty = [t for t in texts if t and t.strip()]
        if not non_empty:
            logger.warning("RAGStore.build_index: all document texts are empty — skipping")
            return
        try:
            self._matrix = self._vectorizer.fit_transform(texts).toarray()
            self._fitted = True
            logger.info(f"RAGStore: indexed {len(self._docs)} documents ({self._matrix.shape[1]} features)")
        except Exception as e:
            logger.error(f"RAGStore.build_index failed: {e}", exc_info=True)

    # ── High-level Indexing Helpers ────────────────────────────────────────────

    def index(
        self,
        fraud_results: Dict[str, Any],
        forecast_results: Dict[str, Any],
        geo_results: Dict[str, Any],
        shops_df=None,
        beneficiaries_df=None,
    ) -> int:
        """
        Convert all agent outputs into text chunks and rebuild the index.
        Returns the number of documents indexed.
        """
        self.reset()

        self._index_fraud(fraud_results)
        self._index_forecasts(forecast_results)
        self._index_geo(geo_results)
        if shops_df is not None and not shops_df.empty:
            self._index_shops(shops_df)
        if beneficiaries_df is not None and not beneficiaries_df.empty:
            self._index_beneficiaries(beneficiaries_df)

        self.build_index()
        return len(self._docs)

    def _index_fraud(self, fraud_results: Dict[str, Any]) -> None:
        summary = fraud_results.get("summary", {})
        # Summary chunk
        self.add_document(RAGDocument(
            doc_id="fraud_summary",
            text=(
                f"Fraud detection summary: {summary.get('total_alerts', 0)} total alerts raised. "
                f"Critical alerts: {summary.get('critical', 0)}. "
                f"High severity: {summary.get('high', 0)}. "
                f"Medium: {summary.get('medium', 0)}. "
                f"Low: {summary.get('low', 0)}. "
                f"Transactions analysed: {summary.get('transactions_analysed', 0)}. "
                f"Fraud rings detected: {summary.get('fraud_rings_detected', 0)}."
            ),
            source="fraud_summary",
            metadata=summary,
        ))

        # Individual critical/high alerts — each as its own chunk
        for alert in fraud_results.get("alerts", []):
            severity = alert.get("severity", "")
            if severity not in ("Critical", "High"):
                continue
            shop = alert.get("fps_shop_id", "unknown shop")
            card = alert.get("beneficiary_card_id", "unknown card")
            pattern = alert.get("fraud_pattern", "unknown pattern")
            score = alert.get("anomaly_score", 0)
            explanation = alert.get("explanation", "")
            self.add_document(RAGDocument(
                doc_id=f"fraud_alert_{alert.get('alert_id', id(alert))}",
                text=(
                    f"Fraud alert ({severity}): shop {shop}, card {card}, "
                    f"pattern '{pattern}', anomaly score {score:.2f}. "
                    f"Explanation: {explanation}"
                ),
                source="fraud_alerts",
                metadata={
                    "severity": severity,
                    "fps_shop_id": shop,
                    "card_id": card,
                    "pattern": pattern,
                    "score": score,
                },
            ))

        # Fraud rings
        for ring in fraud_results.get("fraud_rings", []):
            ring_id = ring.get("ring_id")
            n_cards = ring.get("n_cards", 0)
            n_shops = ring.get("n_shops", 0)
            # ring_report uses "score"; alert dicts use "anomaly_score" — handle both
            score = ring.get("score", ring.get("anomaly_score", 0))
            shop_ids = ", ".join(str(s) for s in (ring.get("shop_ids") or [])[:5])
            card_ids = ", ".join(str(c) for c in (ring.get("card_ids") or [])[:5])
            self.add_document(RAGDocument(
                doc_id=f"fraud_ring_{ring_id}",
                text=(
                    f"Coordinated fraud ring {ring_id}: {n_cards} beneficiary cards "
                    f"linked to {n_shops} FPS shops with ring score {score:.2f}. "
                    f"Shops involved: {shop_ids or 'N/A'}. "
                    f"Sample cards: {card_ids or 'N/A'}. "
                    f"Multi-shop card ratio: {ring.get('multi_shop_ratio', 0):.0%}. "
                    f"Biometric miss rate: {ring.get('bio_miss_rate', 0):.0%}."
                ),
                source="fraud_rings",
                metadata=ring,
            ))

    def _index_forecasts(self, forecast_results: Dict[str, Any]) -> None:
        self.add_document(RAGDocument(
            doc_id="forecast_summary",
            text=(
                f"Demand forecast summary: {forecast_results.get('total_forecasts', 0)} forecasts generated. "
                f"Stock risk flags: {forecast_results.get('total_risk_flags', 0)}."
            ),
            source="forecast_summary",
            metadata={},
        ))
        for flag in forecast_results.get("risk_flags", [])[:20]:
            shop = flag.get("fps_shop_id", "unknown")
            commodity = flag.get("commodity", "unknown")
            flag_type = flag.get("flag_type", "unknown")
            qty = flag.get("forecast_qty_kg", 0)
            self.add_document(RAGDocument(
                doc_id=f"risk_flag_{shop}_{commodity}",
                text=(
                    f"Stock risk flag: shop {shop}, commodity {commodity}, "
                    f"risk type '{flag_type}', forecast quantity {qty:.1f} kg. "
                    f"Recommended action: {flag.get('recommended_action', 'Review stock levels')}."
                ),
                source="forecast_risk_flags",
                metadata=flag,
            ))

    def _index_geo(self, geo_results: Dict[str, Any]) -> None:
        self.add_document(RAGDocument(
            doc_id="geo_summary",
            text=(
                f"Geospatial analysis: overall accessibility score "
                f"{geo_results.get('overall_accessibility_score', 0):.1%}. "
                f"Underserved beneficiaries (>5 km from FPS): {geo_results.get('underserved_count', 0)}. "
                f"New FPS location recommendations: "
                f"{len(geo_results.get('new_location_recommendations', []))}. "
                f"Underperforming shops: {len(geo_results.get('underperforming_shops', []))}."
            ),
            source="geo_summary",
            metadata={},
        ))
        for rec in geo_results.get("new_location_recommendations", [])[:10]:
            district = rec.get("district", "unknown")
            lat = rec.get("latitude", 0)
            lon = rec.get("longitude", 0)
            bene = rec.get("beneficiaries_served", 0)
            self.add_document(RAGDocument(
                doc_id=f"geo_rec_{district}_{lat:.3f}_{lon:.3f}",
                text=(
                    f"New FPS shop recommended in {district} district "
                    f"at coordinates ({lat:.4f}, {lon:.4f}). "
                    f"Would serve approximately {bene} underserved beneficiaries."
                ),
                source="geo_recommendations",
                metadata=rec,
            ))
        for shop in geo_results.get("underperforming_shops", [])[:10]:
            shop_id = shop.get("shop_id", "unknown")
            district = shop.get("district", "unknown")
            reason = shop.get("reason", "low utilisation")
            self.add_document(RAGDocument(
                doc_id=f"underperforming_{shop_id}",
                text=(
                    f"Underperforming FPS shop {shop_id} in {district} district: {reason}."
                ),
                source="geo_underperforming",
                metadata=shop,
            ))

    def _index_shops(self, shops_df) -> None:
        try:
            import pandas as pd
            total = len(shops_df)
            active = int(shops_df["is_active"].sum()) if "is_active" in shops_df.columns else total
            districts = shops_df["district"].nunique() if "district" in shops_df.columns else 0
            self.add_document(RAGDocument(
                doc_id="shops_overview",
                text=(
                    f"FPS shops overview: {total} total shops, {active} active, "
                    f"covering {districts} districts in Telangana."
                ),
                source="shops",
                metadata={"total": total, "active": active, "districts": districts},
            ))
            # Per-district shop counts
            if "district" in shops_df.columns:
                for district, grp in shops_df.groupby("district"):
                    n = len(grp)
                    n_active = int(grp["is_active"].sum()) if "is_active" in grp.columns else n
                    self.add_document(RAGDocument(
                        doc_id=f"shops_district_{district}",
                        text=(
                            f"District {district}: {n} FPS shops total, {n_active} active."
                        ),
                        source="shops_by_district",
                        metadata={"district": district, "total": n, "active": n_active},
                    ))
        except Exception as e:
            logger.warning(f"RAGStore: could not index shops — {e}")

    def _index_beneficiaries(self, beneficiaries_df) -> None:
        try:
            total = len(beneficiaries_df)
            districts = beneficiaries_df["district"].nunique() if "district" in beneficiaries_df.columns else 0
            self.add_document(RAGDocument(
                doc_id="beneficiaries_overview",
                text=(
                    f"Beneficiaries overview: {total} registered beneficiaries "
                    f"across {districts} districts in Telangana."
                ),
                source="beneficiaries",
                metadata={"total": total, "districts": districts},
            ))
        except Exception as e:
            logger.warning(f"RAGStore: could not index beneficiaries — {e}")

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        k: int = 6,
        source_filter: Optional[List[str]] = None,
    ) -> List[Tuple[RAGDocument, float]]:
        """
        Retrieve the top-k documents most relevant to the query.

        Parameters
        ----------
        query : str
            The user's natural language query.
        k : int
            Number of documents to return.
        source_filter : list of str, optional
            If provided, only documents from these source categories are considered.

        Returns
        -------
        List of (RAGDocument, similarity_score) tuples, sorted by score descending.
        """
        if not self._fitted or self._matrix is None or not self._docs:
            logger.warning("RAGStore.retrieve: index not built — returning all docs")
            return [(d, 0.5) for d in self._docs[:k]]

        docs = self._docs
        matrix = self._matrix

        # Apply source filter
        if source_filter:
            indices = [i for i, d in enumerate(docs) if d.source in source_filter]
            if indices:
                docs = [docs[i] for i in indices]
                matrix = matrix[indices]

        if not docs:
            return []

        # Vectorise query
        try:
            q_vec = self._vectorizer.transform([query]).toarray()
        except Exception as e:
            logger.warning(f"RAGStore.retrieve: query vectorisation failed ({e}) — returning top-k by order")
            return [(d, 0.0) for d in docs[:k]]

        # Cosine similarity
        sims = cosine_similarity(q_vec, matrix)[0]
        top_indices = np.argsort(sims)[::-1][:k]

        # Always return top-k — even if all similarities are zero, Claude still needs context
        results = [(docs[i], float(sims[i])) for i in top_indices]

        logger.debug(
            f"RAGStore.retrieve: query='{query[:60]}' → {len(results)} chunks "
            f"(top score={results[0][1]:.3f} source={results[0][0].source})"
            if results else f"RAGStore.retrieve: no results for '{query[:60]}'"
        )
        return results

    def retrieve_as_text(self, query: str, k: int = 6) -> str:
        """Convenience method — returns retrieved chunks as a single formatted string."""
        results = self.retrieve(query, k=k)
        if not results:
            return "No relevant context found in the knowledge base."
        lines = []
        for i, (doc, score) in enumerate(results, 1):
            lines.append(f"[{i}] (source: {doc.source}, relevance: {score:.2f})\n{doc.text}")
        return "\n\n".join(lines)

    # ── Stats ──────────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Return store statistics."""
        sources: Dict[str, int] = {}
        for d in self._docs:
            sources[d.source] = sources.get(d.source, 0) + 1
        return {
            "total_documents": len(self._docs),
            "index_built": self._fitted,
            "documents_by_source": sources,
            "vocab_size": self._matrix.shape[1] if self._matrix is not None else 0,
        }
