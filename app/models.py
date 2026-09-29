"""
SQLAlchemy models for the RUL Fleet Monitoring platform.

Design notes
* `Engine.trajectory_buffer` holds the full CMAPSS trajectory (list of
  dicts). The streamer reveals one entry per tick into `SensorReading`, so
  the app feels like a live fleet instead of an instant dump.
* `SensorReading.sensors` is a JSON blob keyed by `sensor_measure_N`, which
  keeps the schema stable even though CMAPSS has 21 sensors.
* `User.role` follows the hierarchy in `app.rbac`.
* `Alert` and `Prediction` are decoupled so the UI can acknowledge an alert
  without mutating the underlying prediction.
"""
from datetime import datetime

from flask_login import UserMixin

from app import db, login_manager


# User / auth

@login_manager.user_loader
def load_user(user_id: str):
    return User.query.get(int(user_id))


class User(UserMixin, db.Model):
    __tablename__ = "user"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    # viewer | engineer | supervisor | admin
    role = db.Column(db.String(32), default="engineer", nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    notes = db.relationship("Note", backref="author", lazy=True)
    assigned_alerts = db.relationship(
        "Alert",
        backref="assignee",
        lazy=True,
        foreign_keys="Alert.assigned_to",
    )

    def __repr__(self) -> str:
        return f"<User {self.username} ({self.role})>"


# Fleet

class Engine(db.Model):
    __tablename__ = "engine"

    id = db.Column(db.Integer, primary_key=True)
    unit_number = db.Column(db.Integer, nullable=False)
    tag = db.Column(db.String(20), unique=True, nullable=False, index=True)
    dataset = db.Column(db.String(10), nullable=False, default="FD001", index=True)
    # fault mode label, from the CMAPSS paper (FD001 -> HPC, FD003 -> HPC+Fan)
    fault_mode = db.Column(db.String(32), nullable=True)

    # Live state (advanced by the streamer)
    last_cycle = db.Column(db.Integer, default=0)
    last_reading_at = db.Column(db.DateTime)

    # Full CMAPSS trajectory, revealed one step at a time by the streamer.
    # Each entry: {"cycle": int, "op_setting_1": float, ..., "sensors": {...}}
    trajectory_buffer = db.Column(db.JSON)

    # Snapshot of precomputed RUL trend (kept for the detail sparkline).
    rul_trend = db.Column(db.JSON)

    __table_args__ = (
        db.UniqueConstraint("dataset", "unit_number", name="uq_engine_dataset_unit"),
    )

    readings = db.relationship(
        "SensorReading",
        backref="engine",
        lazy="selectin",
        order_by="SensorReading.cycle",
        cascade="all, delete-orphan",
    )
    predictions = db.relationship(
        "Prediction",
        backref="engine",
        lazy="selectin",
        order_by="Prediction.created_at.desc()",
        cascade="all, delete-orphan",
    )
    notes = db.relationship(
        "Note",
        backref="engine",
        lazy=True,
        order_by="Note.created_at.desc()",
        cascade="all, delete-orphan",
    )
    alerts = db.relationship(
        "Alert",
        backref="engine",
        lazy=True,
        cascade="all, delete-orphan",
    )

    @property
    def latest_prediction(self):
        return self.predictions[0] if self.predictions else None

    @property
    def status(self) -> str:
        pred = self.latest_prediction
        if not pred:
            return "unknown"
        if pred.rul <= 20:
            return "critical"
        if pred.rul <= 50:
            return "watch"
        return "healthy"

    def __repr__(self) -> str:
        return f"<Engine {self.tag} {self.dataset}#{self.unit_number}>"


SENSOR_NAMES = [f"sensor_measure_{i}" for i in range(1, 22)]
OP_SETTING_NAMES = ["op_setting_1", "op_setting_2", "op_setting_3"]
ALL_COLUMNS = OP_SETTING_NAMES + SENSOR_NAMES


class SensorReading(db.Model):
    __tablename__ = "sensor_reading"

    id = db.Column(db.Integer, primary_key=True)
    engine_id = db.Column(db.Integer, db.ForeignKey("engine.id"), nullable=False, index=True)
    cycle = db.Column(db.Integer, nullable=False)

    op_setting_1 = db.Column(db.Float)
    op_setting_2 = db.Column(db.Float)
    op_setting_3 = db.Column(db.Float)

    # JSON blob keyed by SENSOR_NAMES to avoid 21 columns.
    sensors = db.Column(db.JSON, nullable=False)

    recorded_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint("engine_id", "cycle", name="uq_reading_engine_cycle"),
    )

    def value(self, col_name: str) -> float:
        if col_name in OP_SETTING_NAMES:
            return getattr(self, col_name, 0.0) or 0.0
        return float(self.sensors.get(col_name, 0.0))



# MLOps

class ModelVersion(db.Model):
    __tablename__ = "model_version"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), unique=True, nullable=False)
    dataset_id = db.Column(db.String(10), nullable=False, index=True)
    path = db.Column(db.String(256))
    window_size = db.Column(db.Integer, default=30)
    trained_on = db.Column(db.String(64))
    val_rmse = db.Column(db.Float)
    cmapss_score = db.Column(db.Float)
    is_active = db.Column(db.Boolean, default=True, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<ModelVersion {self.name} active={self.is_active}>"


class Prediction(db.Model):
    __tablename__ = "prediction"

    id = db.Column(db.Integer, primary_key=True)
    engine_id = db.Column(db.Integer, db.ForeignKey("engine.id"), nullable=False, index=True)
    model_version_id = db.Column(db.Integer, db.ForeignKey("model_version.id"))

    rul = db.Column(db.Integer, nullable=False)
    ci_low = db.Column(db.Integer)
    ci_high = db.Column(db.Integer)
    confidence_label = db.Column(db.String(16))  # low | medium | high

    input_cycle = db.Column(db.Integer)
    top_sensors = db.Column(db.JSON)  # [{"name": "S50", "weight": 0.31}, ...]

    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    model_version = db.relationship("ModelVersion")


# Notes / alerts / audit

TAG_CHOICES = ["Inspection", "Repair", "Anomaly", "Vibration", "Oil", "Handover"]
SEVERITY_CHOICES = ["info", "watch", "critical"]
NOTE_STATUS_CHOICES = ["open", "in_progress", "resolved"]


class Note(db.Model):
    __tablename__ = "note"

    id = db.Column(db.Integer, primary_key=True)
    engine_id = db.Column(db.Integer, db.ForeignKey("engine.id"), nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)

    body = db.Column(db.Text, nullable=False)
    tags = db.Column(db.JSON, default=list)
    severity = db.Column(db.String(16), default="info")
    action_taken = db.Column(db.Text)
    status = db.Column(db.String(16), default="open", index=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Alert(db.Model):
    __tablename__ = "alert"

    id = db.Column(db.Integer, primary_key=True)
    engine_id = db.Column(db.Integer, db.ForeignKey("engine.id"), nullable=False, index=True)
    prediction_id = db.Column(db.Integer, db.ForeignKey("prediction.id"))

    # open | acknowledged | resolved | snoozed
    status = db.Column(db.String(16), default="open", index=True)
    assigned_to = db.Column(db.Integer, db.ForeignKey("user.id"))

    snooze_reason = db.Column(db.String(256))
    snoozed_until = db.Column(db.DateTime)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    resolved_at = db.Column(db.DateTime)

    def __repr__(self) -> str:
        return f"<Alert engine={self.engine_id} status={self.status}>"


class AuditLog(db.Model):
    __tablename__ = "audit_log"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), index=True)
    user = db.relationship("User", foreign_keys=[user_id])
    action = db.Column(db.String(64), index=True)  # note.edit, alert.ack, ...
    target_type = db.Column(db.String(32))
    target_id = db.Column(db.Integer)
    detail = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)