"""
LSTM-based Demand Forecasting Model
Predicts monthly commodity demand per FPS shop using sequential transaction patterns.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
import logging
import os
import json

logger = logging.getLogger(__name__)

# Conditional PyTorch import for environments without GPU
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available — LSTM model will use fallback statistical method")


class LSTMDemandModel(nn.Module if TORCH_AVAILABLE else object):
    """LSTM model for time-series demand forecasting."""

    def __init__(self, input_size: int = 8, hidden_size: int = 64,
                 num_layers: int = 2, output_size: int = 1, dropout: float = 0.2):
        if TORCH_AVAILABLE:
            super(LSTMDemandModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if TORCH_AVAILABLE:
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
            self.fc = nn.Linear(hidden_size, output_size)
            self.relu = nn.ReLU()

    def forward(self, x):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return self.relu(out)


class LSTMForecaster:
    """
    Trains and runs the LSTM forecasting pipeline for a given FPS shop / commodity.
    Falls back to a simple exponential-smoothing approach when PyTorch is unavailable.
    """

    SEQUENCE_LENGTH = 6   # Look back 6 months
    FEATURES = [
        "quantity_lifted",
        "active_cards",
        "month_sin",         # Seasonal encoding
        "month_cos",
        "lag_1",             # Previous month
        "lag_2",
        "rolling_avg_3",
        "rolling_std_3",
    ]

    def __init__(self, model_dir: str = "backend/data/models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.model: Optional[LSTMDemandModel] = None
        self.scaler_params: Dict = {}

    # ── Feature Engineering ───────────────────────────────────────────────────

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().sort_values("month")
        df["month_num"] = pd.to_datetime(df["month"]).dt.month
        df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)
        df["lag_1"] = df["quantity_lifted"].shift(1)
        df["lag_2"] = df["quantity_lifted"].shift(2)
        df["rolling_avg_3"] = df["quantity_lifted"].rolling(3).mean()
        df["rolling_std_3"] = df["quantity_lifted"].rolling(3).std().fillna(0)
        return df.dropna()

    def _normalize(self, data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        mean = data.mean(axis=0)
        std = data.std(axis=0) + 1e-8
        return (data - mean) / std, {"mean": mean.tolist(), "std": std.tolist()}

    def _create_sequences(self, data: np.ndarray, target_col: int = 0):
        X, y = [], []
        for i in range(len(data) - self.SEQUENCE_LENGTH):
            X.append(data[i: i + self.SEQUENCE_LENGTH])
            y.append(data[i + self.SEQUENCE_LENGTH, target_col])
        return np.array(X), np.array(y)

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, transactions_df: pd.DataFrame, epochs: int = 50,
              lr: float = 0.001) -> Dict:
        if not TORCH_AVAILABLE:
            logger.info("PyTorch unavailable — skipping LSTM training, will use fallback")
            return {"status": "skipped", "reason": "pytorch_unavailable"}

        features_df = self._engineer_features(transactions_df)
        if len(features_df) < self.SEQUENCE_LENGTH + 2:
            return {"status": "insufficient_data", "rows": len(features_df)}

        feature_matrix = features_df[self.FEATURES].values
        normalized, self.scaler_params = self._normalize(feature_matrix)
        X, y = self._create_sequences(normalized)

        X_t = torch.FloatTensor(X)
        y_t = torch.FloatTensor(y).unsqueeze(1)

        self.model = LSTMDemandModel(input_size=len(self.FEATURES))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.model.train()
        losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.model(X_t)
            loss = criterion(output, y_t)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        final_loss = losses[-1]
        logger.info(f"LSTM training complete. Final loss: {final_loss:.4f}")
        return {"status": "success", "final_loss": final_loss, "epochs": epochs}

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, transactions_df: pd.DataFrame,
                months_ahead: int = 3) -> List[Dict]:
        features_df = self._engineer_features(transactions_df)

        if not TORCH_AVAILABLE or self.model is None:
            return self._fallback_predict(features_df, months_ahead)

        self.model.eval()
        feature_matrix = features_df[self.FEATURES].values
        mean = np.array(self.scaler_params["mean"])
        std = np.array(self.scaler_params["std"])
        normalized = (feature_matrix - mean) / std

        results = []
        seq = normalized[-self.SEQUENCE_LENGTH:]

        for m in range(months_ahead):
            with torch.no_grad():
                x_tensor = torch.FloatTensor(seq).unsqueeze(0)
                pred_norm = self.model(x_tensor).item()

            # Denormalise
            pred_qty = pred_norm * std[0] + mean[0]
            pred_qty = max(0, pred_qty)

            # 90% CI using ±10% heuristic (replace with MC Dropout for production)
            ci_lower = pred_qty * 0.90
            ci_upper = pred_qty * 1.10

            results.append({
                "month_offset": m + 1,
                "predicted_quantity_kg": round(pred_qty, 2),
                "confidence_lower": round(ci_lower, 2),
                "confidence_upper": round(ci_upper, 2),
                "model": "lstm",
            })

            # Roll the sequence forward with a synthetic next step
            next_step = np.zeros(len(self.FEATURES))
            next_step[0] = pred_norm
            seq = np.vstack([seq[1:], next_step])

        return results

    def _fallback_predict(self, df: pd.DataFrame, months_ahead: int) -> List[Dict]:
        """Exponential smoothing fallback when PyTorch is unavailable."""
        series = df["quantity_lifted"].values
        alpha = 0.3
        smoothed = series[-1]
        for v in series[-6:]:
            smoothed = alpha * v + (1 - alpha) * smoothed

        results = []
        for m in range(months_ahead):
            results.append({
                "month_offset": m + 1,
                "predicted_quantity_kg": round(smoothed, 2),
                "confidence_lower": round(smoothed * 0.85, 2),
                "confidence_upper": round(smoothed * 1.15, 2),
                "model": "exponential_smoothing_fallback",
            })
        return results

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate_mape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        mask = actual != 0
        return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])))

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, shop_id: str, commodity: str):
        path = os.path.join(self.model_dir, f"lstm_{shop_id}_{commodity}")
        if TORCH_AVAILABLE and self.model:
            torch.save(self.model.state_dict(), f"{path}.pt")
        with open(f"{path}_scaler.json", "w") as f:
            json.dump(self.scaler_params, f)

    def load(self, shop_id: str, commodity: str) -> bool:
        path = os.path.join(self.model_dir, f"lstm_{shop_id}_{commodity}")
        scaler_file = f"{path}_scaler.json"
        if not os.path.exists(scaler_file):
            return False
        with open(scaler_file) as f:
            self.scaler_params = json.load(f)
        if TORCH_AVAILABLE:
            model_file = f"{path}.pt"
            if os.path.exists(model_file):
                self.model = LSTMDemandModel(input_size=len(self.FEATURES))
                self.model.load_state_dict(torch.load(model_file, map_location="cpu"))
                self.model.eval()
        return True
