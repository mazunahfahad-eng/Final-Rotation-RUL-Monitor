"""
Prediction orchestration.

Takes a raw sensor window (seq_len × n_features, most recent cycle last) and
returns a dict ready to be persisted as a `Prediction` row, including:
  - point RUL and 80% confidence interval
  - confidence label
  - top sensor drivers for the UI ("what's pushing this reading?")

Uses app.inference.NumpyModel which returns the *scaled* window alongside the
prediction. We reuse that tensor directly for attribution so scaling is
applied exactly once.
"""
from __future__ import annotations

import numpy as np

from app.inference import get_model

# 80% interval half-width from a normal approximation on held-out residuals.
# (rmse in the model config is the closest real error estimate we have.)
Z_80 = 1.2816

# How many recent cycles to compare against the window mean for the plain-LSTM
# drift attribution.
RECENT_WINDOW = 5

# How many top sensors to surface.
TOP_K = 3


# Confidence

def confidence_label(rul: float, half_width: float) -> str:
    """Relative CI width → coarse confidence bucket."""
    ratio = half_width / max(rul, 1.0)
    if ratio < 0.25:
        return "high"
    if ratio < 0.45:
        return "medium"
    return "low"


# Sensor attribution

def _sensor_mask(model) -> np.ndarray:
    """Boolean mask: True for sensor_measure_* columns, False for op_settings."""
    return np.array(
        [c.startswith("sensor_measure_") for c in model.feature_cols],
        dtype=bool,
    )


def _attribution_bilstm_attn(model, scaled_window: np.ndarray,
                            weights: np.ndarray) -> np.ndarray:
    """
    Attention-weighted deviation: |x_t - mean(x)| averaged across timesteps,
    weighted by how much the model attended to each timestep.
    """
    window_mean = scaled_window.mean(axis=0)
    dev = np.abs(scaled_window - window_mean)          # (seq, n_features)
    weighted = (weights[:, None] * dev).sum(axis=0)    # (n_features,)
    return weighted


def _attribution_lstm_plain(model, scaled_window: np.ndarray) -> np.ndarray:
    """
    No attention → fall back to recent-cycle drift vs. the whole window mean.
    Larger drift on a feature == stronger contribution to the recent change.
    """
    recent = scaled_window[-RECENT_WINDOW:].mean(axis=0)
    overall = scaled_window.mean(axis=0)
    return np.abs(recent - overall)


def _top_sensors(model, contributions: np.ndarray) -> list[dict]:
    """Return up to TOP_K sensor drivers, named like the UI expects."""
    mask = _sensor_mask(model)
    # Rank all features, then keep sensor columns only.
    order = np.argsort(contributions)[::-1]
    keep = [i for i in order if mask[i]][:TOP_K]

    total = float(sum(contributions[i] for i in keep)) or 1.0
    out = []
    for i in keep:
        # "sensor_measure_4" -> "S4"
        pretty = model.feature_cols[i].replace("sensor_measure_", "S")
        out.append({
            "name": pretty,
            "weight": round(float(contributions[i] / total), 2),
        })
    return out


# Public API

def predict_for_window(dataset_id: str, raw_matrix: np.ndarray) -> dict:
    """
    raw_matrix : (seq_len, n_features) raw sensor values, columns in
                 model.feature_cols order, most recent cycle last.

    Returns a dict ready to feed into `Prediction(...)`. The model itself is
    resolved from the currently active ModelVersion for `dataset_id`.
    """
    model = get_model(dataset_id)

    rul, weights, scaled_window = model.predict_window(raw_matrix)

    rmse = float(model.config["test_metrics"]["rmse"])
    half_width = Z_80 * rmse
    ci_low = max(0, int(round(rul - half_width)))
    ci_high = int(round(rul + half_width))

    if weights is not None:
        contrib = _attribution_bilstm_attn(model, scaled_window, weights)
    else:
        contrib = _attribution_lstm_plain(model, scaled_window)

    return {
        "rul": int(round(rul)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "confidence_label": confidence_label(rul, half_width),
        "top_sensors": _top_sensors(model, contrib),
    }