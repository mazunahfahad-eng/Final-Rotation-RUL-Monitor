"""
HTTP layer.

Sections
1. Dashboard & engine pages         (view_* capabilities)
2. Notes                            (add/edit/delete, own vs any)
3. Alerts                           (ack / snooze / resolve / assign)
4. Model registry (MLOps)           (list / promote / rollback)
5. JSON APIs
     /api/summary        KPI counts + last-update timestamp
     /api/engines        fleet list (filterable by dataset/status)
     /api/history/<id>   per-engine series, RUL trend + CI, anomalies, drivers
     /api/predict/<id>   latest prediction payload

Every mutating route writes an AuditLog row so the supervisor view has a
trail. RBAC is enforced by @require(capability) from app.rbac.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from flask import (
    Blueprint, abort, jsonify, redirect, render_template,
    request, url_for,
)
from flask_login import current_user, login_required

from app import db
from app.models import (
    Alert, AuditLog, Engine, ModelVersion, Note, Prediction, User,
    NOTE_STATUS_CHOICES, SEVERITY_CHOICES, TAG_CHOICES,
)
from app.rbac import require

main_bp = Blueprint("main", __name__)


# Helpers

def _audit(action: str, target_type: str, target_id: int, detail: str = "") -> None:
    db.session.add(AuditLog(
        user_id=current_user.id,
        action=action,
        target_type=target_type,
        target_id=target_id,
        detail=detail[:500],
    ))


def _filtered_engines() -> list[Engine]:
    q = Engine.query
    dataset = request.args.get("dataset", "").strip().upper()
    if dataset in ("FD001", "FD003"):
        q = q.filter(Engine.dataset == dataset)

    status = request.args.get("status", "").strip()
    engines = q.all()
    if status:
        engines = [e for e in engines if e.status == status]

    search = request.args.get("q", "").strip().lower()
    if search:
        engines = [e for e in engines if search in (e.tag or "").lower()]

    # Sort ascending by RUL — worst first, which is what an operator wants.
    engines.sort(key=lambda e: e.latest_prediction.rul if e.latest_prediction else 9999)
    return engines


# 1. Dashboard & engine detail

@main_bp.route("/")
@login_required
@require("view_dashboard")
def dashboard():
    engines = _filtered_engines()
    # Counts reflect *all* engines (not just filtered), so the KPI strip
    # always shows the true fleet health.
    counts = {"critical": 0, "watch": 0, "healthy": 0, "unknown": 0}
    for e in Engine.query.all():
        counts[e.status] = counts.get(e.status, 0) + 1

    last_update = (
        Engine.query.order_by(Engine.last_reading_at.desc()).first()
    )
    last_update = last_update.last_reading_at if last_update and last_update.last_reading_at else None

    return render_template(
        "dashboard.html",
        engines=engines,
        counts=counts,
        q=request.args.get("q", ""),
        status_filter=request.args.get("status", ""),
        dataset_filter=request.args.get("dataset", ""),
        last_update=last_update,
    )


@main_bp.route("/engine/<int:engine_id>")
@login_required
@require("view_engine")
def engine_detail(engine_id: int):
    engine = Engine.query.get_or_404(engine_id)
    tab = request.args.get("tab", "overview")
    return render_template(
        "engine_detail.html",
        engine=engine,
        tab=tab,
        tag_choices=TAG_CHOICES,
        severity_choices=SEVERITY_CHOICES,
        note_status_choices=NOTE_STATUS_CHOICES,
    )


# 2. Notes
@main_bp.route("/engine/<int:engine_id>/notes", methods=["POST"])
@login_required
@require("add_note")
def add_note(engine_id: int):
    engine = Engine.query.get_or_404(engine_id)
    body = request.form.get("body", "").strip()
    if not body:
        return jsonify({"error": "Note can't be empty."}), 400

    note = Note(
        engine_id=engine.id,
        user_id=current_user.id,
        body=body,
        tags=request.form.getlist("tags"),
        severity=request.form.get("severity", "info"),
        action_taken=(request.form.get("action_taken", "").strip() or None),
        status="open",
    )
    db.session.add(note)
    db.session.flush()
    _audit("note.create", "note", note.id, body)
    db.session.commit()
    return render_template("_note_card.html", note=note)


@main_bp.route("/notes/<int:note_id>", methods=["PUT", "DELETE"])
@login_required
def note_detail(note_id: int):
    note = Note.query.get_or_404(note_id)
    is_owner = note.user_id == current_user.id

    if request.method == "DELETE":
        cap = "delete_own_note" if is_owner else "delete_any_note"
        if not _can(cap):
            abort(403)
        _audit("note.delete", "note", note.id, note.body[:200])
        db.session.delete(note)
        db.session.commit()
        return "", 204

    # PUT
    cap = "edit_own_note" if is_owner else "edit_any_note"
    if not _can(cap):
        abort(403)

    new_body = request.form.get("body", note.body).strip() or note.body
    note.body = new_body
    note.severity = request.form.get("severity", note.severity)
    note.status = request.form.get("status", note.status)
    note.updated_at = datetime.utcnow()
    _audit("note.edit", "note", note.id)
    db.session.commit()
    return render_template("_note_card.html", note=note)


# 3. Alerts

@main_bp.route("/alerts")
@login_required
@require("view_alerts")
def alerts():
    sort = request.args.get("sort", "rul")
    if sort not in ("rul", "oldest", "mine"):
        sort = "rul"

    q = (
        Alert.query
        .filter(Alert.status.in_(["open", "acknowledged", "snoozed"]))
        .join(Engine)
    )
    if sort == "mine":
        q = q.filter(Alert.assigned_to == current_user.id)
    open_alerts = q.all()

    if sort == "oldest":
        open_alerts.sort(key=lambda a: a.created_at or datetime.utcnow())
    else:
        open_alerts.sort(
            key=lambda a: a.engine.latest_prediction.rul if a.engine.latest_prediction else 9999
        )

    engineers = User.query.filter_by(role="engineer").order_by(User.username).all()
    return render_template("alerts.html", alerts=open_alerts, engineers=engineers, sort=sort)


@main_bp.route("/alerts/<int:alert_id>/ack", methods=["POST"])
@login_required
@require("ack_own_alert")
def ack_alert(alert_id: int):
    alert = Alert.query.get_or_404(alert_id)
    if not _can("ack_alert") and alert.assigned_to != current_user.id:
        abort(403)  # engineers may only acknowledge alerts assigned to them

    if alert.status == "open":
        alert.status = "acknowledged"
        if alert.assigned_to is None:
            alert.assigned_to = current_user.id
        _audit("alert.ack", "alert", alert.id, f"by={current_user.username}")
        db.session.commit()
    return render_template("_alert_row.html", alert=alert, engineers=User.query.filter_by(role="engineer").order_by(User.username).all())


@main_bp.route("/alerts/<int:alert_id>/snooze", methods=["POST"])
@login_required
@require("snooze_alert")
def snooze_alert(alert_id: int):
    alert = Alert.query.get_or_404(alert_id)
    reason = request.form.get("reason", "").strip()
    if not reason:
        return jsonify({"error": "Give a reason before snoozing."}), 400

    hours = int(request.form.get("hours", "24") or 24)
    alert.status = "snoozed"
    alert.snooze_reason = reason
    alert.snoozed_until = datetime.utcnow() + timedelta(hours=hours)
    _audit("alert.snooze", "alert", alert.id, reason)
    db.session.commit()
    return render_template("_alert_row.html", alert=alert, engineers=User.query.filter_by(role="engineer").order_by(User.username).all())


@main_bp.route("/alerts/<int:alert_id>/resolve", methods=["POST"])
@login_required
@require("resolve_alert")
def resolve_alert(alert_id: int):
    alert = Alert.query.get_or_404(alert_id)
    alert.status = "resolved"
    alert.resolved_at = datetime.utcnow()
    _audit("alert.resolve", "alert", alert.id)
    db.session.commit()
    return render_template("_alert_row.html", alert=alert, engineers=User.query.filter_by(role="engineer").order_by(User.username).all())


@main_bp.route("/alerts/<int:alert_id>/assign", methods=["POST"])
@login_required
@require("assign_alert")
def assign_alert(alert_id: int):
    alert = Alert.query.get_or_404(alert_id)
    user_id = request.form.get("user_id", type=int)
    engineer = User.query.filter_by(id=user_id, role="engineer").first() if user_id else None
    if engineer is None:
        return jsonify({"error": "Select a valid engineer."}), 400

    if alert.assigned_to != engineer.id:
        alert.assigned_to = engineer.id
        if alert.status == "acknowledged":
            alert.status = "open" 

    _audit("alert.assign", "alert", alert.id, f"user_id={engineer.id}")
    db.session.commit()
    return render_template("_alert_row.html", alert=alert, engineers=User.query.filter_by(role="engineer").order_by(User.username).all())


@main_bp.route("/audit")
@login_required
@require("view_audit")
def audit_page():
    logs = AuditLog.query.order_by(AuditLog.created_at.desc(), AuditLog.id.desc()).all()
    return render_template("audit.html", logs=logs)


# 4. Model registry (MLOps)

@main_bp.route("/models")
@login_required
@require("view_models")
def models_page():
    versions = ModelVersion.query.order_by(ModelVersion.created_at.desc()).all()
    active = [v for v in versions if v.is_active]

    # Pre-compute the active RMSE per dataset so the template doesn't have to
    # guess by regex-matching the free-form `trained_on` string. If a dataset
    # has no active version, the value is None and the template renders "—".
    rmse_by_dataset: dict[str, float | None] = {}
    for ds in ("FD001", "FD003"):
        match = next((v for v in active if v.dataset_id == ds), None)
        rmse_by_dataset[ds] = match.val_rmse if match else None

    rollback_targets = set()
    for ds in ("FD001", "FD003"):
        ds_versions = [v for v in versions if v.dataset_id == ds]
        if len(ds_versions) > 1:
            current = next((v for v in ds_versions if v.is_active), None)
            if current is not None:
                rollback_targets.add(current.id)

    return render_template(
        "models.html",
        versions=versions,
        active_versions=active,
        rmse_by_dataset=rmse_by_dataset,
        rollback_targets=rollback_targets,
    )


@main_bp.route("/models/<int:model_id>/promote", methods=["POST"])
@login_required
@require("promote_model")
def promote_model(model_id: int):
    selected = ModelVersion.query.get_or_404(model_id)
    if selected.is_active:
        return redirect(url_for("main.models_page"))

    # Keep exactly one active version for the dataset.
    (ModelVersion.query
        .filter_by(dataset_id=selected.dataset_id)
        .update({"is_active": False}))
    selected.is_active = True
    _audit("model.promote", "model_version", selected.id, selected.name)
    db.session.commit()
    return redirect(url_for("main.models_page"))


@main_bp.route("/models/<int:model_id>/rollback", methods=["POST"])
@login_required
@require("rollback_model")
def rollback_model(model_id: int):
    current = ModelVersion.query.get_or_404(model_id)
    if not current.is_active:
        return redirect(url_for("main.models_page"))

    versions = (
        ModelVersion.query
        .filter_by(dataset_id=current.dataset_id)
        .order_by(ModelVersion.created_at.desc(), ModelVersion.id.desc())
        .all()
    )
    try:
        index = next(i for i, version in enumerate(versions) if version.id == current.id)
    except StopIteration:
        return redirect(url_for("main.models_page"))

    if index + 1 >= len(versions):
        return redirect(url_for("main.models_page"))

    previous = versions[index + 1]
    (ModelVersion.query
        .filter_by(dataset_id=current.dataset_id)
        .update({"is_active": False}))
    previous.is_active = True
    _audit(
        "model.rollback",
        "model_version",
        previous.id,
        f"from={current.name} to={previous.name}",
    )
    db.session.commit()
    return redirect(url_for("main.models_page"))


# 5. JSON APIs

@main_bp.route("/api/summary")
@login_required
@require("view_dashboard")
def api_summary():
    counts = {"critical": 0, "watch": 0, "healthy": 0, "unknown": 0}
    for e in Engine.query.all():
        counts[e.status] = counts.get(e.status, 0) + 1

    last = Engine.query.order_by(Engine.last_reading_at.desc()).first()
    streamer = request.app.extensions.get("streamer") if hasattr(request, "app") else None
    streamer = streamer or (request.environ.get("flask.app") and None)  # type: ignore

    from flask import current_app
    streamer = current_app.extensions.get("streamer")

    payload = {
        "counts": counts,
        "last_update": (last.last_reading_at.isoformat() + "Z") if last and last.last_reading_at else None,
    }
    if streamer is not None:
        payload["streamer"] = {
            "ticks": streamer.ticks,
            "reveals": streamer.reveals,
            "predictions": streamer.predictions,
            "alive": bool(streamer._thread and streamer._thread.is_alive()),
        }
    return jsonify(payload)


@main_bp.route("/api/engines")
@login_required
@require("view_dashboard")
def api_engines():
    out = []
    for e in _filtered_engines():
        p = e.latest_prediction
        out.append({
            "id": e.id,
            "tag": e.tag,
            "dataset": e.dataset,
            "status": e.status,
            "rul": p.rul if p else None,
            "ci": [p.ci_low, p.ci_high] if p else None,
            "confidence": p.confidence_label if p else None,
            "last_cycle": e.last_cycle,
            "last_reading_at": (e.last_reading_at.isoformat() + "Z") if e.last_reading_at else None,
        })
    return jsonify(out)


@main_bp.route("/api/history/<int:engine_id>")
@login_required
@require("view_engine")
def api_history(engine_id: int):
    engine = Engine.query.get_or_404(engine_id)
    readings = engine.readings  # ordered by cycle

    # choose which sensors to chart
    default_sensors = [
        "sensor_measure_2", "sensor_measure_3", "sensor_measure_4",
        "sensor_measure_7", "sensor_measure_11", "sensor_measure_12",
    ]
    p = engine.latest_prediction
    flagged: list[str] = []
    if p and p.top_sensors:
        for s in p.top_sensors:
            name = s.get("name", "")
            if name.startswith("S") and name[1:].isdigit():
                flagged.append(f"sensor_measure_{name[1:]}")
    sensor_names = flagged + [s for s in default_sensors if s not in flagged]
    sensor_names = sensor_names[:4]

    cycles = [r.cycle for r in readings]
    series = {name: [r.sensors.get(name) for r in readings] for name in sensor_names}

    # RUL trend with CI from every stored prediction
    preds = engine.predictions  # desc order
    rul_trend = [
        {
            "cycle": pr.input_cycle,
            "rul": pr.rul,
            "ci_low": pr.ci_low,
            "ci_high": pr.ci_high,
            "confidence": pr.confidence_label,
        }
        for pr in reversed(preds)
    ]

    # anomaly detection on the charted sensors
    # z-score vs a rolling mean/std; |z| > 3 flagged as anomalies.
    anomalies = []
    for name in sensor_names:
        vals = np.asarray([r.sensors.get(name, np.nan) for r in readings], dtype=float)
        if vals.size < 10 or np.all(np.isnan(vals)):
            continue
        s = pd.Series(vals)
        roll_mean = s.rolling(10, min_periods=3).mean()
        roll_std = s.rolling(10, min_periods=3).std().replace(0, np.nan)
        z = (s - roll_mean) / roll_std
        for i, zi in enumerate(z.to_numpy()):
            if np.isfinite(zi) and abs(zi) > 3:
                anomalies.append({
                    "cycle": cycles[i],
                    "sensor": name,
                    "z": round(float(zi), 2),
                })

    # drivers (from latest prediction)
    drivers = (p.top_sensors if p and p.top_sensors else [])

    return jsonify({
        "engine": {
            "id": engine.id,
            "tag": engine.tag,
            "dataset": engine.dataset,
            "status": engine.status,
            "last_cycle": engine.last_cycle,
        },
        "cycles": cycles,
        "series": series,
        "rul_trend": rul_trend,
        "anomalies": anomalies,
        "drivers": drivers,
        "thresholds": {"critical": 20, "watch": 50, "cap": 125},
    })


@main_bp.route("/api/predict/<int:engine_id>")
@login_required
@require("view_engine")
def api_predict(engine_id: int):
    engine = Engine.query.get_or_404(engine_id)
    p = engine.latest_prediction
    if not p:
        return jsonify({"error": "No prediction yet for this engine."}), 404
    return jsonify({
        "rul": p.rul,
        "ci_low": p.ci_low,
        "ci_high": p.ci_high,
        "confidence": p.confidence_label,
        "top_sensors": p.top_sensors,
        "model_version": p.model_version.name if p.model_version else None,
        "input_cycle": p.input_cycle,
        "created_at": p.created_at.isoformat() + "Z",
    })


# Small RBAC helper (imported lazily to avoid circular import in some setups)

def _can(capability: str) -> bool:
    from app.rbac import can
    return can(capability)