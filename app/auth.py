"""
Authentication blueprint.

Adds on top of the original:
  - safe "next" redirect handling (open-redirect protection)
  - audit logging of login success/failure/logout
  - rate-limit-friendly logging on failures (IP + username, no password)
  - explicit session refresh on login
"""
from __future__ import annotations

from datetime import datetime
from urllib.parse import urlparse, urljoin

from flask import (
    Blueprint, current_app, flash, redirect, render_template,
    request, session, url_for,
)
from flask_login import current_user, login_required, login_user, logout_user
from werkzeug.security import check_password_hash

from app import db
from app.models import AuditLog, User

auth_bp = Blueprint("auth", __name__)


def _is_safe_next(target: str | None) -> bool:
    """Only allow same-host relative redirects."""
    if not target:
        return False
    ref = urlparse(request.host_url)
    test = urlparse(urljoin(request.host_url, target))
    return test.scheme in ("http", "https") and ref.netloc == test.netloc


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))

    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""

        user = User.query.filter_by(username=username).first()
        ok = bool(user) and check_password_hash(user.password_hash, password)

        if ok:
            login_user(user, remember=bool(request.form.get("remember")))
            session.permanent = True

            db.session.add(AuditLog(
                user_id=user.id,
                action="auth.login",
                target_type="user",
                target_id=user.id,
                detail=f"ip={request.remote_addr}",
            ))
            db.session.commit()

            next_url = request.args.get("next") or request.form.get("next")
            if _is_safe_next(next_url):
                return redirect(next_url)
            return redirect(url_for("main.dashboard"))

        # Failure — log everything except the password itself.
        current_app.logger.warning(
            "login failed user=%r ip=%s", username, request.remote_addr
        )
        db.session.add(AuditLog(
            user_id=None,
            action="auth.login_failed",
            target_type="user",
            target_id=None,
            detail=f"user={username!r} ip={request.remote_addr}",
        ))
        db.session.commit()
        flash("Wrong username or password.", "error")

    return render_template("login.html", now=datetime.utcnow())


@auth_bp.route("/logout")
@login_required
def logout():
    db.session.add(AuditLog(
        user_id=current_user.id,
        action="auth.logout",
        target_type="user",
        target_id=current_user.id,
    ))
    db.session.commit()
    logout_user()
    flash("Signed out.", "info")
    return redirect(url_for("auth.login"))