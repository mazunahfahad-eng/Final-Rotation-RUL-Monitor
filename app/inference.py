"""
Numpy-only inference for the two trained candidate models.

  FD001 -- AttentionBiLSTMRegressor (2-layer, bidirectional, hidden=96)
  FD003 -- LSTMRegressor           (2-layer, unidirectional, hidden=96)

Weights come from train/train.py (JAX training, numpy export). No JAX or
torch is needed at serve time -- this is what actually runs in the Flask app.

Fixes vs. the previous version
* `predict_window` now returns the *scaled* window it actually consumed, so
  callers (see app/predictor.py) never re-scale the same data. The old code
  double-scaled for the attention attribution step, which distorted the
  top-sensor weights.
* LayerNorm is applied over the full feature vector exactly like training
  (mean/var over the whole vector, not per-batch).
* Model config is validated at load time so a mismatched feature list or
  missing key fails fast.
* The in-process model cache is thread-safe (Flask serves with gthread /
  gunicorn workers, so plain dicts aren't safe).
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from app.models import ModelVersion

MODELS_DIR = Path(__file__).parent.parent / "models_bin"

# Cap on softmax to avoid overflow when scores are large.
_SOFTMAX_CLIP = 60.0


# Small numeric helpers

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def lstm_direction(w: dict, x: np.ndarray, hidden: int, reverse: bool = False) -> np.ndarray:
    """Run one LSTM direction. x: (seq, in_dim) -> (seq, hidden)."""
    seq = x[::-1] if reverse else x
    h = np.zeros(hidden, dtype=np.float32)
    c = np.zeros(hidden, dtype=np.float32)
    outs = np.zeros((seq.shape[0], hidden), dtype=np.float32)

    W_ih, W_hh = w["W_ih"], w["W_hh"]
    b_ih, b_hh = w["b_ih"], w["b_hh"]

    for t in range(seq.shape[0]):
        gates = seq[t] @ W_ih.T + b_ih + h @ W_hh.T + b_hh
        i, f, g, o = np.split(gates, 4)
        i, f, o = sigmoid(i), sigmoid(f), sigmoid(o)
        g = np.tanh(g)
        c = f * c + i * g
        h = o * np.tanh(c)
        outs[t] = h

    if reverse:
        outs = outs[::-1]
    return outs


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Mean/var over the whole vector — matches train/train.py exactly."""
    mean = x.mean()
    var = x.var()
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def _softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.clip(scores, -_SOFTMAX_CLIP, _SOFTMAX_CLIP)
    scores = scores - scores.max()
    exp = np.exp(scores)
    return exp / (exp.sum() + 1e-12)


# Numpy model

REQUIRED_CONFIG_KEYS = {
    "dataset_id", "model_kind", "feature_cols", "seq_len", "hidden",
    "cap", "feat_min", "feat_max", "y_min", "y_max",
    "test_metrics", "trained_on",
}


class NumpyModel:
    """
    Load once, call `predict_window` many times.

    Returns a (rul, weights, scaled_window) triple so downstream code
    (attribution, diagnostics) can reuse the exact tensor that was scored.
    `weights` is None for the plain-LSTM model.
    """

    def __init__(self, version: ModelVersion):
        dataset_id = version.dataset_id.upper()
        self.model_version_id = int(version.id)
        self.dataset_id = dataset_id

        config_path = MODELS_DIR / f"{dataset_id.lower()}_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Model config not found: {config_path}")

        with config_path.open() as fh:
            self.config: dict = json.load(fh)

        missing = REQUIRED_CONFIG_KEYS - set(self.config)
        if missing:
            raise ValueError(f"{config_path} missing keys: {sorted(missing)}")

        self.model_kind: str = self.config["model_kind"]
        weights_path = Path(version.path) if version.path else MODELS_DIR / f"{dataset_id.lower()}_{self.model_kind}.npz"
        if not weights_path.is_absolute():
            weights_path = MODELS_DIR.parent / weights_path
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")

        with np.load(weights_path) as raw:
            self.w: dict = self._unflatten(raw)

        self.hidden: int = int(self.config["hidden"])
        self.seq_len: int = int(self.config["seq_len"])
        self.feature_cols: list[str] = list(self.config["feature_cols"])

        # Feature scaler, in feature_cols order.
        self.feat_min = np.asarray(self.config["feat_min"], dtype=np.float32)
        self.feat_max = np.asarray(self.config["feat_max"], dtype=np.float32)
        if self.feat_min.shape != (len(self.feature_cols),):
            raise ValueError("feat_min length does not match feature_cols")
        if self.feat_max.shape != (len(self.feature_cols),):
            raise ValueError("feat_max length does not match feature_cols")

        span = self.feat_max - self.feat_min
        self.feat_range = np.where(span < 1e-8, 1.0, span).astype(np.float32)

        # Target scaler.
        self.y_min = float(self.config["y_min"])
        self.y_max = float(self.config["y_max"])
        self.y_range = max(self.y_max - self.y_min, 1e-8)

    # Weight tree unpacking: "head.W1" -> {"head": {"W1": ...}}

    @staticmethod
    def _unflatten(raw: "np.lib.npyio.NpzFile") -> dict:
        tree: dict = {}
        for key in raw.files:
            parts = key.split(".")
            node = tree
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = raw[key]
        return tree

    # Public API

    def scale_features(self, raw_matrix: np.ndarray) -> np.ndarray:
        return (raw_matrix - self.feat_min) / self.feat_range

    def unscale_target(self, scaled_value: float) -> float:
        return scaled_value * self.y_range + self.y_min

    def predict_window(self, raw_matrix: np.ndarray):
        """
        raw_matrix: (seq_len, n_features) raw sensor values, most recent last,
                    columns ordered exactly like `self.feature_cols`.

        Returns: (rul, attention_weights_or_None, scaled_window)
        """
        raw_matrix = np.asarray(raw_matrix, dtype=np.float32)
        if raw_matrix.ndim != 2 or raw_matrix.shape[1] != len(self.feature_cols):
            raise ValueError(
                f"expected shape (seq_len, {len(self.feature_cols)}), "
                f"got {raw_matrix.shape}"
            )

        x_scaled = self.scale_features(raw_matrix)

        if self.model_kind == "bilstm_attn":
            pred_scaled, weights = self._forward_bilstm_attn(x_scaled)
        elif self.model_kind == "lstm_plain":
            pred_scaled, weights = self._forward_lstm_plain(x_scaled)
        else:
            raise ValueError(f"Unknown model_kind: {self.model_kind}")

        rul = self.unscale_target(pred_scaled)
        return max(0.0, float(rul)), weights, x_scaled

    # Forward passes

    def _forward_bilstm_attn(self, x_scaled: np.ndarray):
        w = self.w
        o1f = lstm_direction(w["l1_fwd"], x_scaled, self.hidden, reverse=False)
        o1b = lstm_direction(w["l1_bwd"], x_scaled, self.hidden, reverse=True)
        l1_out = np.concatenate([o1f, o1b], axis=-1)

        o2f = lstm_direction(w["l2_fwd"], l1_out, self.hidden, reverse=False)
        o2b = lstm_direction(w["l2_bwd"], l1_out, self.hidden, reverse=True)
        lstm_out = np.concatenate([o2f, o2b], axis=-1)  # (seq, 2H)

        scores = (
            np.tanh(lstm_out @ w["attn_W1"].T + w["attn_b1"])
            @ w["attn_W2"].T + w["attn_b2"]
        )[:, 0]
        weights = _softmax(scores)
        context = (weights[:, None] * lstm_out).sum(axis=0)

        h = layer_norm(context, w["head"]["ln_gamma"], w["head"]["ln_beta"])
        h = h @ w["head"]["W1"].T + w["head"]["b1"]
        h = np.maximum(h, 0)
        out = h @ w["head"]["W2"].T + w["head"]["b2"]
        return float(out[0]), weights.astype(np.float32)

    def _forward_lstm_plain(self, x_scaled: np.ndarray):
        w = self.w
        o1 = lstm_direction(w["l1"], x_scaled, self.hidden, reverse=False)
        o2 = lstm_direction(w["l2"], o1, self.hidden, reverse=False)
        last = o2[-1]
        h = layer_norm(last, w["head"]["ln_gamma"], w["head"]["ln_beta"])
        out = h @ w["head"]["W"].T + w["head"]["b"]
        return float(out[0]), None


# Thread-safe cache

# Cache by dataset, but retain the ModelVersion id that produced the object.
# Every get_model() call re-resolves the active DB row so promotion/rollback
# takes effect without requiring a process restart.
_models: dict[str, tuple[int, NumpyModel]] = {}
_models_lock = threading.Lock()


def _active_version(dataset_id: str) -> ModelVersion:
    key = dataset_id.upper()
    version = (
        ModelVersion.query
        .filter_by(dataset_id=key, is_active=True)
        .order_by(ModelVersion.created_at.desc(), ModelVersion.id.desc())
        .first()
    )
    if version is None:
        raise RuntimeError(f"No active model version for dataset {key}")
    return version


def get_model(dataset_id: str) -> NumpyModel:
    """Resolve and load the active ModelVersion for a dataset.

    The active version is checked on every call. A cached model is reused only
    when its ModelVersion id still matches the active DB row.
    """
    key = dataset_id.upper()
    active = _active_version(key)
    cached = _models.get(key)
    if cached is not None and cached[0] == active.id:
        return cached[1]

    with _models_lock:
        # Re-check after acquiring the lock because another request/thread may
        # have populated the cache while we waited.
        active = _active_version(key)
        cached = _models.get(key)
        if cached is not None and cached[0] == active.id:
            return cached[1]
        model = NumpyModel(active)
        _models[key] = (active.id, model)
        return model


def loaded_models() -> list[str]:
    """Used by /readyz."""
    return sorted(_models.keys())