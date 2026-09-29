"""
Application factory for the RUL Fleet Monitoring platform.

Wires:
  - SQLAlchemy + Flask-Login
  - RBAC capability context processor
  - Blueprints: main, auth, health
  - Structured JSON logging
  - Graceful DB creation/migration
  - Optional background telemetry streamer (disabled in tests)
"""
import json
import logging
import os
import sys

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager


db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = "auth.login"
login_manager.login_message = "Please sign in to continue."
login_manager.login_message_category = "info"


class JsonFormatter(logging.Formatter):
    """One JSON object per log line — friendly to log shippers (Loki/ELK)."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def _configure_logging(app: Flask) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    # Avoid duplicate handlers under the Flask reloader.
    if not any(isinstance(h, logging.StreamHandler) and isinstance(h.formatter, JsonFormatter)
               for h in root.handlers):
        root.handlers = [handler]
    root.setLevel(logging.INFO)
    app.logger.setLevel(logging.INFO)


def _default_sqlite_uri(app: Flask) -> str:
    return "sqlite:///" + os.path.join(app.instance_path, "rul.db")


def _ensure_model_version_schema() -> None:
    """Bring older demo databases up to the active-model schema."""
    from sqlalchemy import inspect, text

    inspector = inspect(db.engine)
    if "model_version" not in inspector.get_table_names():
        return

    columns = {c["name"] for c in inspector.get_columns("model_version")}
    if "dataset_id" not in columns:
        db.session.execute(text(
            "ALTER TABLE model_version ADD COLUMN dataset_id VARCHAR(10)"
        ))
        db.session.execute(text(
            "UPDATE model_version SET dataset_id = CASE "
            "WHEN upper(trained_on) LIKE 'FD001%' THEN 'FD001' "
            "WHEN upper(trained_on) LIKE 'FD003%' THEN 'FD003' "
            "END"
        ))
        db.session.commit()

    # Repair legacy rows before the unique active-version index is created.
    rows = db.session.execute(text(
        "SELECT id, dataset_id FROM model_version "
        "WHERE is_active = 1 ORDER BY dataset_id, created_at DESC, id DESC"
    )).all()
    seen = set()
    for row in rows:
        dataset_id = row.dataset_id
        if dataset_id in seen:
            db.session.execute(
                text("UPDATE model_version SET is_active = 0 WHERE id = :id"),
                {"id": row.id},
            )
        else:
            seen.add(dataset_id)
    db.session.commit()

    db.session.execute(text(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_model_version_active_dataset "
        "ON model_version(dataset_id) WHERE is_active = 1"
    ))
    db.session.commit()


def create_app(config_overrides: dict | None = None) -> Flask:
    app = Flask(__name__, instance_relative_config=True)

    # config
    app.config.update(
        SECRET_KEY=os.environ.get("SECRET_KEY", "dev-only-change-me"),
        SQLALCHEMY_DATABASE_URI=os.environ.get("DATABASE_URL") or _default_sqlite_uri(app),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        SQLALCHEMY_ENGINE_OPTIONS={"pool_pre_ping": True},
        ENABLE_STREAMER=os.environ.get("ENABLE_STREAMER", "1") == "1",
        STREAM_TICK=int(os.environ.get("STREAM_TICK", "8")),
        STREAM_BATCH=int(os.environ.get("STREAM_BATCH", "1")),
        STREAM_PREDICT_EVERY=int(os.environ.get("STREAM_PREDICT_EVERY", "5")),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        REMEMBER_COOKIE_HTTPONLY=True,
    )
    if config_overrides:
        app.config.update(config_overrides)

    os.makedirs(app.instance_path, exist_ok=True)

    _configure_logging(app)

    # extensions
    db.init_app(app)
    login_manager.init_app(app)

    # RBAC helpers available in every template
    from app.rbac import can, ROLE_RANK, CAPABILITIES  # noqa: E402

    @app.context_processor
    def _inject_rbac():
        return {
            "can": can,
            "ROLE_RANK": ROLE_RANK,
            "CAPABILITIES": CAPABILITIES,
        }

    # blueprints
    from app.routes import main_bp
    from app.auth import auth_bp
    from app.health import health_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(health_bp)

    # schema
    with app.app_context():
        from app import models  # noqa: F401 — register models with SQLAlchemy
        db.create_all()
        _ensure_model_version_schema()

    # background streamer
    # Disabled automatically in tests (TESTING=True) or when ENABLE_STREAMER=0.
    if app.config["ENABLE_STREAMER"] and not app.config.get("TESTING"):
        try:
            from app.streamer import FleetStreamer, should_run_streamer

            if should_run_streamer():
                streamer = FleetStreamer(app)
                streamer.start()
                app.extensions["streamer"] = streamer
                app.logger.info(
                    "Fleet streamer started (pid=%d, tick=%ss, batch=%s, predict_every=%s)",
                    os.getpid(),
                    app.config["STREAM_TICK"],
                    app.config["STREAM_BATCH"],
                    app.config["STREAM_PREDICT_EVERY"],
                )
            else:
                app.logger.info(
                    "Streamer not started on this worker (pid=%d, "
                    "STREAMER_LEADER=%r). Expected under multi-worker gunicorn; "
                    "exactly one worker should have STREAMER_LEADER=1.",
                    os.getpid(),
                    os.environ.get("STREAMER_LEADER"),
                )
        except Exception as exc:
            app.logger.exception("Failed to start fleet streamer: %s", exc)

    return app