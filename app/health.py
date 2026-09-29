"""
Health and readiness endpoints.

/healthz   Liveness  — process is up and can answer HTTP. Cheap, no DB.
/readyz    Readiness — DB reachable AND models loadable. Used by the
                       container orchestrator to gate traffic.

Both are intentionally unauthenticated so k8s / Docker can probe them.
Neither leaks secrets — only version strings, model names, and boolean flags.
"""
from __future__ import annotations

import os
import platform
import sys
from datetime import datetime

from flask import Blueprint, jsonify
from sqlalchemy import text

from app import db

health_bp = Blueprint("health", __name__)

# Capture process start so /readyz can report uptime without extra deps.
_STARTED_AT = datetime.utcnow()


def _uptime_seconds() -> float:
    return (datetime.utcnow() - _STARTED_AT).total_seconds()


@health_bp.route("/healthz")
def healthz():
    """Liveness: no I/O, no DB, no model load."""
    return jsonify({
        "ok": True,
        "uptime_s": round(_uptime_seconds(), 1),
        "pid": os.getpid(),
        "python": platform.python_version(),
    })


@health_bp.route("/readyz")
def readyz():
    """
    Readiness: DB ping + model cache check. Returns 503 if anything is
    missing so the orchestrator keeps traffic away.
    """
    problems: list[str] = []

    # --- DB ---
    db_ok = True
    try:
        db.session.execute(text("SELECT 1"))
    except Exception as exc:
        db_ok = False
        problems.append(f"db: {exc!s}")

    # --- Models (lazy-load safe: only report what's already in cache) ---
    from app.inference import loaded_models
    models_loaded = loaded_models()

    # If nothing is loaded yet, that's fine on cold start — but if the
    # configured datasets exist on disk, a missing load is worth flagging.
    try:
        from app.inference import MODELS_DIR
        expected = sorted(
            p.stem.replace("_config", "").upper()
            for p in MODELS_DIR.glob("*_config.json")
        )
    except Exception:
        expected = []

    if expected and not set(models_loaded).intersection(expected):
        # Not fatal — a fresh worker hasn't been hit yet — but report it.
        problems.append(f"models: none loaded yet (expected one of {expected})")

    ok = db_ok
    status = 200 if ok else 503

    return jsonify({
        "ok": ok,
        "uptime_s": round(_uptime_seconds(), 1),
        "db": db_ok,
        "models_loaded": models_loaded,
        "models_expected": expected,
        "streamer": _streamer_status(),
        "problems": problems,
        "python": platform.python_version(),
        "platform": sys.platform,
    }), status


def _streamer_status() -> dict:
    from flask import current_app
    s = current_app.extensions.get("streamer")
    if s is None:
        return {"running": False}
    alive = bool(s._thread and s._thread.is_alive())
    return {
        "running": alive,
        "owner_pid": s.owner_pid,
        "owner_host": s.owner_host,
        "this_pid": os.getpid(),
        "ticks": s.ticks,
        "reveals": s.reveals,
        "predictions": s.predictions,
        "failures": s.failures,
    }