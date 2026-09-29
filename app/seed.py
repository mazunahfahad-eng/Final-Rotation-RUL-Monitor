"""
Seed the fleet.

What this does

1. Wipes and recreates the schema.
2. Creates four users, one per role in `app.rbac`: viewer / engineer /
   supervisor / admin.
3. Registers one `ModelVersion` per trained candidate (FD001 BiLSTM+Attn,
   FD003 plain LSTM) with its real metrics from the model config.
4. For each engine in the *test* splits of FD001 and FD003:
     - stores the full trajectory in `Engine.trajectory_buffer`
     - writes ONLY the first `seq_len` readings as SensorReading rows
     - runs one inference to produce a starting prediction and alert
       if warranted
5. Leaves the rest of the trajectory to be revealed live by the streamer.
   Notes are NOT seeded — they are real operator input, added from the UI.

FD003 fault modes
Per the C-MAPSS paper:
  FD003 units 1-100 contain a mix of HPC degradation and Fan degradation.
  The published split is 50/50 across the 100 test units, interleaved.
We tag each engine with its fault mode from a deterministic index pattern
so the rotation can demonstrate "cluster by fault mode".

Run
---
    python -m app.seed
"""
from __future__ import annotations

import json
import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from werkzeug.security import generate_password_hash

from app import create_app, db
from app.inference import get_model
from app.models import (
    Alert, Engine, ModelVersion, Prediction, SensorReading, User,
)
from app.predictor import predict_for_window



# Config

DATA_DIR = os.environ.get("SEED_DATA_DIR", "../data")
BASE_COLUMNS = ["unit_number", "time_cycles", "op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLUMNS = [f"sensor_measure_{i}" for i in range(1, 22)]

# Demo password for every seeded account. Override in prod.
DEMO_PASSWORD = os.environ.get("SEED_PASSWORD", "password")



# Loaders

def load_frame(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    n_sensors = len(df.columns) - len(BASE_COLUMNS)
    df.columns = BASE_COLUMNS + [f"sensor_measure_{i}" for i in range(1, n_sensors + 1)]
    return df


def load_rul(dataset_id: str) -> pd.Series:
    path = f"{DATA_DIR}/RUL_{dataset_id}.txt"
    rul = pd.read_csv(path, sep=r"\s+", header=None, names=["RUL"])
    rul.index = rul.index + 1
    rul.index.name = "unit_number"
    return rul["RUL"]



# FD003 fault-mode assignment

def fd003_fault_mode(unit_number: int) -> str:
    """
    Deterministic 50/50 interleave of the two FD003 fault modes.
    Odd units → HPC degradation; even units → Fan degradation.
    (The published test set does not ship per-unit labels, so this is a
    plausible rotation-friendly labelling for demonstration only.)
    """
    return "HPC" if unit_number % 2 == 1 else "Fan"


def fault_mode_for(dataset_id: str, unit_number: int) -> str | None:
    if dataset_id == "FD001":
        return "HPC"
    if dataset_id == "FD003":
        return fd003_fault_mode(unit_number)
    return None



# Users + model versions

def seed_users() -> dict[str, User]:
    users = {
        "viewer": User(
            username="viewer",
            role="viewer",
            password_hash=generate_password_hash(DEMO_PASSWORD),
        ),
        "engineer": User(
            username="engineer",
            role="engineer",
            password_hash=generate_password_hash(DEMO_PASSWORD),
        ),
        "supervisor": User(
            username="supervisor",
            role="supervisor",
            password_hash=generate_password_hash(DEMO_PASSWORD),
        ),
        "admin": User(
            username="admin",
            role="admin",
            password_hash=generate_password_hash(DEMO_PASSWORD),
        ),
    }
    db.session.add_all(users.values())
    db.session.commit()
    return users


def seed_model_versions() -> dict[str, ModelVersion]:
    def load_config(dataset_id: str) -> dict:
        with open(f"models_bin/{dataset_id.lower()}_config.json", encoding="utf-8") as fh:
            return json.load(fh)

    fd001_cfg = load_config("FD001")
    fd003_cfg = load_config("FD003")

    mv1 = ModelVersion(
        name="attn-bilstm-fd001",
        dataset_id="FD001",
        path="models_bin/fd001_bilstm_attn.npz",
        window_size=int(fd001_cfg["seq_len"]),
        trained_on=fd001_cfg["trained_on"],
        val_rmse=float(fd001_cfg["test_metrics"]["rmse"]),
        cmapss_score=float(fd001_cfg["test_metrics"]["cmapss_score"]),
        is_active=True,
    )
    mv2 = ModelVersion(
        name="lstm-fd003",
        dataset_id="FD003",
        path="models_bin/fd003_lstm_plain.npz",
        window_size=int(fd003_cfg["seq_len"]),
        trained_on=fd003_cfg["trained_on"],
        val_rmse=float(fd003_cfg["test_metrics"]["rmse"]),
        cmapss_score=float(fd003_cfg["test_metrics"]["cmapss_score"]),
        is_active=True,
    )
    db.session.add_all([mv1, mv2])
    db.session.commit()
    return {"FD001": mv1, "FD003": mv2}



# Fleet seeding

def _build_reading_entry(row: pd.Series) -> dict:
    return {
        "cycle": int(row["time_cycles"]),
        "op_setting_1": float(row["op_setting_1"]),
        "op_setting_2": float(row["op_setting_2"]),
        "op_setting_3": float(row["op_setting_3"]),
        "sensors": {f"sensor_measure_{j}": float(row[f"sensor_measure_{j}"])
                    for j in range(1, 22)},
    }


def _seed_one_engine(
    dataset_id: str,
    unit: int,
    rows: pd.DataFrame,
    model,
    model_version: ModelVersion,
    users: dict[str, User],
    now: datetime,
) -> Engine:
    rows = rows.sort_values("time_cycles").reset_index(drop=True)
    last_cycle = int(rows["time_cycles"].max())

    engine = Engine(
        unit_number=int(unit),
        tag=f"EG-{dataset_id[-1]}{unit:03d}",
        dataset=dataset_id,
        fault_mode=fault_mode_for(dataset_id, int(unit)),
        last_cycle=int(rows["time_cycles"].iloc[0]),  # start-of-life cursor
        last_reading_at=now - timedelta(hours=random.randint(0, 6)),
        trajectory_buffer=None,
    )
    db.session.add(engine)
    db.session.flush()

    # Full trajectory as JSON buffer (unrevealed future).
    buffer = [_build_reading_entry(r) for _, r in rows.iterrows()]
    engine.trajectory_buffer = buffer

    # Reveal only the first seq_len readings now (so the model can score).
    warmup = buffer[:model.seq_len]
    for entry in warmup:
        db.session.add(SensorReading(
            engine_id=engine.id,
            cycle=entry["cycle"],
            op_setting_1=entry["op_setting_1"],
            op_setting_2=entry["op_setting_2"],
            op_setting_3=entry["op_setting_3"],
            sensors=entry["sensors"],
        ))
    engine.last_cycle = int(warmup[-1]["cycle"])
    engine.trajectory_buffer = buffer[model.seq_len:]

    # Warm-up inference on the initial window.
    raw_matrix = rows.iloc[:model.seq_len][model.feature_cols].values
    pred = predict_for_window(dataset_id, raw_matrix.astype(np.float32))

    prediction = Prediction(
        engine_id=engine.id,
        model_version_id=model_version.id,
        rul=pred["rul"],
        ci_low=pred["ci_low"],
        ci_high=pred["ci_high"],
        confidence_label=pred["confidence_label"],
        input_cycle=engine.last_cycle,
        top_sensors=pred["top_sensors"],
        created_at=now - timedelta(hours=random.randint(0, 4)),
    )
    db.session.add(prediction)
    db.session.flush()

    if pred["rul"] <= 20:
        db.session.add(Alert(engine_id=engine.id, prediction_id=prediction.id, status="open"))

    # Rough precomputed trend so the detail sparkline has something to show
    # before the streamer produces enough live points.
    engine.rul_trend = [{
        "cycle": engine.last_cycle,
        "rul": pred["rul"],
        "ci_low": pred["ci_low"],
        "ci_high": pred["ci_high"],
    }]

    return engine


def seed_dataset(
    dataset_id: str,
    model_versions: dict[str, ModelVersion],
    users: dict[str, User],
    now: datetime,
) -> None:
    model = get_model(dataset_id)
    test_df = load_frame(f"{DATA_DIR}/test_{dataset_id}.txt")

    unit_ids = sorted(int(u) for u in test_df["unit_number"].unique())
    print(f"[{dataset_id}] seeding {len(unit_ids)} engines "
          f"(warmup {model.seq_len} cycles, rest streamed)")

    for i, unit in enumerate(unit_ids):
        rows = test_df[test_df["unit_number"] == unit]
        _seed_one_engine(
            dataset_id=dataset_id,
            unit=unit,
            rows=rows,
            model=model,
            model_version=model_versions[dataset_id],
            users=users,
            now=now,
        )
        if (i + 1) % 25 == 0:
            db.session.commit()
            print(f"  {i + 1}/{len(unit_ids)} committed")

    db.session.commit()



# Entry point

def main() -> None:
    random.seed(42)
    np.random.seed(42)

    app = create_app({"ENABLE_STREAMER": False, "TESTING": True})
    with app.app_context():
        print("Wiping schema...")
        db.drop_all()
        db.create_all()

        print("Seeding users...")
        users = seed_users()

        print("Registering model versions...")
        model_versions = seed_model_versions()

        now = datetime.utcnow()
        seed_dataset("FD001", model_versions, users, now)
        seed_dataset("FD003", model_versions, users, now)

        engines = Engine.query.count()
        readings = SensorReading.query.count()
        preds = Prediction.query.count()

        print()
        print(f"Fleet seeded: {engines} engines")
        print(f"  initial readings: {readings}")
        print(f"  initial predictions: {preds}")
        print(f"  remaining cycles will stream live (ENABLE_STREAMER=1)")
        print()
        print("Login with any of:")
        for role in ("viewer", "engineer", "supervisor", "admin"):
            print(f"  {role:10s} / {DEMO_PASSWORD}")


if __name__ == "__main__":
    main()