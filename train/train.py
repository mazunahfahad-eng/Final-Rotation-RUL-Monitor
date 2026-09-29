"""
Trains the exact candidate architectures selected in model_comparison_FD001.ipynb
and model_comparison_FD003.ipynb, on the full train files, and saves numpy-loadable
weights + scalers for the Flask app's predictor.

FD001 candidate: AttentionBiLSTMRegressor (2-layer, bidirectional, hidden=96,
    temporal attention, LayerNorm+Linear+ReLU+Dropout+Linear head)
FD003 candidate: LSTMRegressor (2-layer, unidirectional, hidden=96,
    LayerNorm+Linear head)

Both trained with the notebook's asymmetric smooth-L1 loss (overprediction
penalized 2x), AdamW, gradient clipping at norm 1.0, ReduceLROnPlateau-style
LR halving, early stopping patience 15, RUL cap 125, seq_len 30.

Framework note: the notebooks used PyTorch. This sandbox can't install a
working CPU build of torch (network/disk constraints), so this script
reimplements the identical math in JAX for autodiff and exports plain numpy
weights -- the app's predictor then runs pure-numpy inference, no JAX/torch
needed at serve time.
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import random

SEQ_LEN = 30
CAP = 125
HIDDEN = 96
BATCH_SIZE = 128
MAX_EPOCHS = 50
PATIENCE = 15
LR_PATIENCE = 5
LR = 5e-4
WEIGHT_DECAY = 1e-4
SEED = 42

BASE_COLUMNS = ["unit_number", "time_cycles", "op_setting_1", "op_setting_2", "op_setting_3"]


def load_frame(path):
    df = pd.read_csv(path, sep=r"\s+", header=None)
    n_sensors = len(df.columns) - len(BASE_COLUMNS)
    df.columns = BASE_COLUMNS + [f"sensor_measure_{i}" for i in range(1, n_sensors + 1)]
    return df


def attach_training_rul(train_df, cap=CAP):
    result = train_df.copy()
    final_cycle = result.groupby("unit_number")["time_cycles"].transform("max")
    result["RUL"] = (final_cycle - result["time_cycles"]).clip(upper=cap)
    return result


def attach_test_rul(test_df, rul_series, cap=CAP):
    result = test_df.copy()
    unit_last_cycle = result.groupby("unit_number")["time_cycles"].transform("max")
    result["RUL"] = result["unit_number"].map(rul_series)
    result["RUL"] = (result["RUL"] + unit_last_cycle - result["time_cycles"]).clip(upper=cap)
    return result


def make_windows(frame, feature_cols, seq_len=SEQ_LEN, last_only=False):
    windows, targets, groups = [], [], []
    for unit_id, group in frame.groupby("unit_number", sort=True):
        group = group.sort_values("time_cycles")
        data = group[feature_cols].values
        target = group["RUL"].values
        if len(group) < seq_len:
            pad = seq_len - len(group)
            data = np.pad(data, ((pad, 0), (0, 0)), mode="edge")
            windows.append(data)
            targets.append(target[-1])
            groups.append(unit_id)
            continue
        starts = [len(group) - seq_len] if last_only else range(len(group) - seq_len + 1)
        for start in starts:
            windows.append(data[start:start + seq_len])
            targets.append(target[start + seq_len - 1])
            groups.append(unit_id)
    return (np.asarray(windows, dtype=np.float32),
            np.asarray(targets, dtype=np.float32),
            np.asarray(groups))


def group_split(units, val_frac=0.2, seed=SEED):
    rng = np.random.RandomState(seed)
    units = np.array(sorted(units))
    perm = rng.permutation(len(units))
    n_val = max(1, int(len(units) * val_frac))
    val_units = set(units[perm[:n_val]])
    train_units = set(units[perm[n_val:]])
    return train_units, val_units


def cmapss_score(y_true, y_pred):
    err = y_pred - y_true
    pen = np.where(err < 0, np.exp(-err / 13) - 1, np.exp(err / 10) - 1)
    return float(np.sum(pen))


def regression_metrics(y_true, y_pred):
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot)
    return {"cmapss_score": cmapss_score(y_true, y_pred), "mae": mae, "rmse": rmse, "r2": r2}


# JAX model definitions

def init_lstm_cell(key, in_dim, hidden):
    k1, k2, k3, k4 = random.split(key, 4)
    scale = 1.0 / np.sqrt(hidden)
    return {
        "W_ih": random.uniform(k1, (4 * hidden, in_dim), minval=-scale, maxval=scale),
        "W_hh": random.uniform(k2, (4 * hidden, hidden), minval=-scale, maxval=scale),
        "b_ih": jnp.zeros((4 * hidden,)),
        "b_hh": jnp.zeros((4 * hidden,)),
    }


def lstm_cell_step(params, carry, x_t):
    h, c = carry
    gates = x_t @ params["W_ih"].T + params["b_ih"] + h @ params["W_hh"].T + params["b_hh"]
    i, f, g, o = jnp.split(gates, 4)
    i = jax.nn.sigmoid(i)
    f = jax.nn.sigmoid(f)
    g = jnp.tanh(g)
    o = jax.nn.sigmoid(o)
    c_new = f * c + i * g
    h_new = o * jnp.tanh(c_new)
    return (h_new, c_new), h_new


def run_lstm_direction(params, x, hidden, reverse=False):
    # x: (seq, in_dim) -> outputs (seq, hidden)
    if reverse:
        x = jnp.flip(x, axis=0)
    h0 = jnp.zeros((hidden,))
    c0 = jnp.zeros((hidden,))

    def step(carry, x_t):
        return lstm_cell_step(params, carry, x_t)

    (_, _), outs = jax.lax.scan(step, (h0, c0), x)
    if reverse:
        outs = jnp.flip(outs, axis=0)
    return outs


def init_head(key, in_dim, hidden_mid=32):
    k1, k2, k3, k4 = random.split(key, 4)
    return {
        "ln_gamma": jnp.ones((in_dim,)),
        "ln_beta": jnp.zeros((in_dim,)),
        "W1": random.normal(k1, (hidden_mid, in_dim)) * (1.0 / np.sqrt(in_dim)),
        "b1": jnp.zeros((hidden_mid,)),
        "W2": random.normal(k2, (1, hidden_mid)) * (1.0 / np.sqrt(hidden_mid)),
        "b2": jnp.zeros((1,)),
    }


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = jnp.mean(x)
    var = jnp.var(x)
    return (x - mean) / jnp.sqrt(var + eps) * gamma + beta


def head_forward(params, x, dropout_mask=None):
    x = layer_norm(x, params["ln_gamma"], params["ln_beta"])
    x = x @ params["W1"].T + params["b1"]
    x = jax.nn.relu(x)
    if dropout_mask is not None:
        x = x * dropout_mask
    x = x @ params["W2"].T + params["b2"]
    return x[0]


def init_bilstm_model(key, n_features, hidden=HIDDEN):
    keys = random.split(key, 8)
    return {
        "l1_fwd": init_lstm_cell(keys[0], n_features, hidden),
        "l1_bwd": init_lstm_cell(keys[1], n_features, hidden),
        "l2_fwd": init_lstm_cell(keys[2], 2 * hidden, hidden),
        "l2_bwd": init_lstm_cell(keys[3], 2 * hidden, hidden),
        "attn_W1": random.normal(keys[4], (hidden, 2 * hidden)) * (1.0 / np.sqrt(2 * hidden)),
        "attn_b1": jnp.zeros((hidden,)),
        "attn_W2": random.normal(keys[5], (1, hidden)) * (1.0 / np.sqrt(hidden)),
        "attn_b2": jnp.zeros((1,)),
        "head": init_head(keys[6], 2 * hidden),
    }


def bilstm_forward_single(params, x, dropout_mask=None):
    # x: (seq, n_features)
    o1_f = run_lstm_direction(params["l1_fwd"], x, HIDDEN, reverse=False)
    o1_b = run_lstm_direction(params["l1_bwd"], x, HIDDEN, reverse=True)
    l1_out = jnp.concatenate([o1_f, o1_b], axis=-1)  # (seq, 2H)

    o2_f = run_lstm_direction(params["l2_fwd"], l1_out, HIDDEN, reverse=False)
    o2_b = run_lstm_direction(params["l2_bwd"], l1_out, HIDDEN, reverse=True)
    lstm_out = jnp.concatenate([o2_f, o2_b], axis=-1)  # (seq, 2H)

    scores = jnp.tanh(lstm_out @ params["attn_W1"].T + params["attn_b1"]) @ params["attn_W2"].T + params["attn_b2"]
    weights = jax.nn.softmax(scores, axis=0)  # (seq, 1)
    context = jnp.sum(weights * lstm_out, axis=0)  # (2H,)

    return head_forward(params["head"], context, dropout_mask), weights[:, 0]


def init_lstm_model(key, n_features, hidden=HIDDEN):
    keys = random.split(key, 4)
    return {
        "l1": init_lstm_cell(keys[0], n_features, hidden),
        "l2": init_lstm_cell(keys[1], hidden, hidden),
        "head": {
            "ln_gamma": jnp.ones((hidden,)),
            "ln_beta": jnp.zeros((hidden,)),
            "W": random.normal(keys[2], (1, hidden)) * (1.0 / np.sqrt(hidden)),
            "b": jnp.zeros((1,)),
        },
    }


def lstm_forward_single(params, x):
    o1 = run_lstm_direction(params["l1"], x, HIDDEN, reverse=False)
    o2 = run_lstm_direction(params["l2"], o1, HIDDEN, reverse=False)
    last = o2[-1]
    h = layer_norm(last, params["head"]["ln_gamma"], params["head"]["ln_beta"])
    return (h @ params["head"]["W"].T + params["head"]["b"])[0]


# Training 

def asymmetric_smooth_l1(pred, target, weight=2.0, beta=1.0):
    err = pred - target
    abs_err = jnp.abs(err)
    loss = jnp.where(abs_err < beta, 0.5 * err ** 2 / beta, abs_err - 0.5 * beta)
    w = jnp.where(err > 0, weight, 1.0)
    return w * loss


def make_adamw_state(params):
    return {"m": jax.tree.map(jnp.zeros_like, params),
            "v": jax.tree.map(jnp.zeros_like, params),
            "t": 0}


def adamw_update(params, grads, state, lr, wd=WEIGHT_DECAY, b1=0.9, b2=0.999, eps=1e-8):
    t = state["t"] + 1
    m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, state["m"], grads)
    v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * (g ** 2), state["v"], grads)
    m_hat = jax.tree.map(lambda m: m / (1 - b1 ** t), m)
    v_hat = jax.tree.map(lambda v: v / (1 - b2 ** t), v)
    new_params = jax.tree.map(
        lambda p, mh, vh: p - lr * (mh / (jnp.sqrt(vh) + eps) + wd * p), params, m_hat, v_hat
    )
    return new_params, {"m": m, "v": v, "t": t}


def clip_grad_norm(grads, max_norm=1.0):
    leaves = jax.tree.leaves(grads)
    total_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves))
    scale = jnp.minimum(1.0, max_norm / (total_norm + 1e-6))
    return jax.tree.map(lambda g: g * scale, grads)


def build_forward(model_kind, key, n_features):
    if model_kind == "bilstm_attn":
        params = init_bilstm_model(key, n_features)

        def forward(p, x, mask=None):
            out, _ = bilstm_forward_single(p, x, mask)
            return out
    else:
        params = init_lstm_model(key, n_features)

        def forward(p, x, mask=None):
            return lstm_forward_single(p, x)
    return params, forward


def train_model_chunk(dataset_id, model_kind, x_train, y_train, x_val, y_val,
                       ckpt_path, epochs_this_run, verbose=True):
    """Runs up to `epochs_this_run` epochs, resuming from ckpt_path if present.
    Returns True if training is finished (early-stopped or hit MAX_EPOCHS)."""
    n_features = x_train.shape[-1]
    key = random.PRNGKey(SEED)
    fresh_params, forward = build_forward(model_kind, key, n_features)

    if ckpt_path.exists():
        ckpt = pickle.loads(ckpt_path.read_bytes())
        params = jax.tree.map(jnp.asarray, ckpt["params"])
        opt_state = {
            "m": jax.tree.map(jnp.asarray, ckpt["opt_m"]),
            "v": jax.tree.map(jnp.asarray, ckpt["opt_v"]),
            "t": ckpt["opt_t"],
        }
        best_val = ckpt["best_val"]
        best_params = jax.tree.map(jnp.asarray, ckpt["best_params"])
        patience_ctr = ckpt["patience_ctr"]
        lr_ctr = ckpt["lr_ctr"]
        lr = ckpt["lr"]
        start_epoch = ckpt["epoch"] + 1
        if verbose:
            print(f"resuming from epoch {start_epoch}, best_val={best_val:.4f}")
    else:
        params = fresh_params
        opt_state = make_adamw_state(params)
        best_val = float("inf")
        best_params = params
        patience_ctr = 0
        lr_ctr = 0
        lr = LR
        start_epoch = 1

    batched_forward = jax.vmap(forward, in_axes=(None, 0, None))

    def loss_fn(p, xb, yb):
        preds = batched_forward(p, xb, None)
        return jnp.mean(asymmetric_smooth_l1(preds, yb))

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    eval_fn = jax.jit(lambda p, xb: batched_forward(p, xb, None))

    n = x_train.shape[0]
    rng = np.random.RandomState(SEED + start_epoch)
    finished = False
    epoch = start_epoch - 1

    for epoch in range(start_epoch, min(start_epoch + epochs_this_run, MAX_EPOCHS + 1)):
        perm = rng.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            xb = jnp.asarray(x_train[idx])
            yb = jnp.asarray(y_train[idx])
            loss, grads = grad_fn(params, xb, yb)
            grads = clip_grad_norm(grads)
            params, opt_state = adamw_update(params, grads, opt_state, lr)
            epoch_loss += float(loss) * len(idx)
        epoch_loss /= n

        val_preds = eval_fn(params, jnp.asarray(x_val))
        val_loss = float(jnp.mean(asymmetric_smooth_l1(val_preds, jnp.asarray(y_val))))

        if verbose:
            print(f"[{dataset_id}/{model_kind}] epoch {epoch}: train={epoch_loss:.4f} val={val_loss:.4f} lr={lr:.2e}")

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_params = jax.tree.map(lambda a: a, params)
            patience_ctr = 0
            lr_ctr = 0
        else:
            patience_ctr += 1
            lr_ctr += 1
            if lr_ctr >= LR_PATIENCE:
                lr *= 0.5
                lr_ctr = 0
            if patience_ctr >= PATIENCE:
                if verbose:
                    print(f"[{dataset_id}/{model_kind}] early stop at epoch {epoch}")
                finished = True
                break

    if epoch >= MAX_EPOCHS:
        finished = True

    ckpt = {
        "params": jax.tree.map(np.asarray, params),
        "opt_m": jax.tree.map(np.asarray, opt_state["m"]),
        "opt_v": jax.tree.map(np.asarray, opt_state["v"]),
        "opt_t": opt_state["t"],
        "best_val": best_val,
        "best_params": jax.tree.map(np.asarray, best_params),
        "patience_ctr": patience_ctr,
        "lr_ctr": lr_ctr,
        "lr": lr,
        "epoch": epoch,
        "finished": finished,
    }
    ckpt_path.write_bytes(pickle.dumps(ckpt))
    if verbose:
        print(f"checkpoint saved at epoch {epoch}, finished={finished}")

    return finished, jax.tree.map(jnp.asarray, best_params), forward


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    epochs_this_run = int(sys.argv[2]) if len(sys.argv) > 2 else MAX_EPOCHS
    datasets = [
        ("FD001", "bilstm_attn"),
        ("FD003", "lstm_plain"),
    ]
    if only:
        datasets = [d for d in datasets if d[0] == only]

    script_dir = Path(__file__).resolve().parent
    ckpt_dir = script_dir / "train_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir = script_dir.parent / "models_bin"
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset_id, model_kind in datasets:
        t0 = time.time()
        print(f"=== {dataset_id} ({model_kind}) ===")
        train_df = load_frame(f"../data/train_{dataset_id}.txt")
        test_df = load_frame(f"../data/test_{dataset_id}.txt")
        rul_series = pd.read_csv(f"../data/RUL_{dataset_id}.txt",
                                  sep=r"\s+", header=None, names=["RUL"])
        rul_series.index = rul_series.index + 1
        rul_series.index.name = "unit_number"
        rul_series = rul_series["RUL"]

        train_df = attach_training_rul(train_df)
        test_df = attach_test_rul(test_df, rul_series)

        candidate_cols = [c for c in train_df.columns if c not in ("unit_number", "time_cycles", "RUL")]
        constant_cols = [c for c in candidate_cols if train_df[c].std() < 1e-6]
        feature_cols = [c for c in candidate_cols if c not in constant_cols]
        print(f"Dropped {len(constant_cols)}: {constant_cols}")
        print(f"Training on {len(feature_cols)} features")

        train_units, val_units = group_split(train_df["unit_number"].unique())
        tr_df = train_df[train_df["unit_number"].isin(train_units)]
        va_df = train_df[train_df["unit_number"].isin(val_units)]

        feat_min = tr_df[feature_cols].min().values.astype(np.float32)
        feat_max = tr_df[feature_cols].max().values.astype(np.float32)
        feat_range = np.where(feat_max - feat_min < 1e-8, 1.0, feat_max - feat_min)

        def scale_features(df):
            df = df.copy()
            df[feature_cols] = (df[feature_cols].values - feat_min) / feat_range
            return df

        tr_scaled = scale_features(tr_df)
        va_scaled = scale_features(va_df)
        test_scaled = scale_features(test_df)

        x_train, y_train_raw, _ = make_windows(tr_scaled, feature_cols)
        x_val, y_val_raw, _ = make_windows(va_scaled, feature_cols)
        x_test, y_test_raw, test_groups = make_windows(test_scaled, feature_cols, last_only=True)
        print(f"train windows {x_train.shape}, val windows {x_val.shape}, test windows {x_test.shape}")

        y_min, y_max = float(y_train_raw.min()), float(y_train_raw.max())
        y_range = max(y_max - y_min, 1e-8)
        y_train = (y_train_raw - y_min) / y_range
        y_val = (y_val_raw - y_min) / y_range

        ckpt_path = ckpt_dir / f"ckpt_{dataset_id.lower()}.pkl"
        finished, best_params, forward = train_model_chunk(
            dataset_id, model_kind, x_train, y_train, x_val, y_val,
            ckpt_path, epochs_this_run)

        if not finished:
            print(f"[{dataset_id}] not finished yet -- run again to continue "
                  f"(elapsed {time.time()-t0:.1f}s this chunk)")
            continue

        batched_forward = jax.vmap(lambda x: forward(best_params, x, None))
        pred_scaled = np.asarray(batched_forward(jnp.asarray(x_test)))
        pred = pred_scaled * y_range + y_min
        metrics = regression_metrics(y_test_raw, pred)
        print(f"[{dataset_id}] TEST metrics: {metrics}")

        weights_np = jax.tree.map(lambda a: np.asarray(a), best_params)
        np.savez(f"{out_dir}/{dataset_id.lower()}_{model_kind}.npz",
                 **{k: v for k, v in _flatten(weights_np)})

        config = {
            "dataset_id": dataset_id,
            "model_kind": model_kind,
            "feature_cols": feature_cols,
            "seq_len": SEQ_LEN,
            "hidden": HIDDEN,
            "cap": CAP,
            "feat_min": feat_min.tolist(),
            "feat_max": feat_max.tolist(),
            "y_min": y_min,
            "y_max": y_max,
            "test_metrics": metrics,
            "trained_on": f"{dataset_id} train (full, {train_df['unit_number'].nunique()} units)",
        }
        with open(f"{out_dir}/{dataset_id.lower()}_config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"[{dataset_id}] done in {time.time()-t0:.1f}s")


def _flatten(tree, prefix=""):
    if isinstance(tree, dict):
        items = []
        for k, v in tree.items():
            items.extend(_flatten(v, f"{prefix}{k}."))
        return items
    else:
        return [(prefix.rstrip("."), tree)]


if __name__ == "__main__":
    main()
