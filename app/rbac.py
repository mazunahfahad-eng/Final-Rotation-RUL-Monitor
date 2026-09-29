"""
Role-Based Access Control for the RUL Fleet Monitoring platform.

Four roles, strict hierarchy:

    viewer < engineer < supervisor < admin

Permissions are expressed as *capabilities* (verbs), so callers never
string-compare roles. Templates and routes both use the same `can()` helper,
which keeps the UI and the API in sync automatically.

Usage in a route:
    from app.rbac import require
    @main_bp.route("/alerts/<int:id>/ack", methods=["POST"])
    @login_required
    @require("ack_alert")
    def ack_alert(id): ...

Usage in a template:
    {% if can('ack_alert') %}<button>Acknowledge</button>{% endif %}
"""
from functools import wraps

from flask import abort, current_app
from flask_login import current_user


# Numeric rank makes comparisons cheap and total.
ROLE_RANK: dict[str, int] = {
    "viewer": 0,
    "engineer": 1,
    "supervisor": 2,
    "admin": 3,
}


# Capability -> minimum role required to exercise it.
# Order is irrelevant; look-ups are O(1).
CAPABILITIES: dict[str, str] = {
    # Read-only
    "view_dashboard":     "viewer",
    "view_engine":        "viewer",
    "view_alerts":        "viewer",
    "view_audit":         "admin",

    # Notes
    "add_note":           "engineer",
    "edit_own_note":      "engineer",
    "edit_any_note":      "supervisor",
    "delete_own_note":    "engineer",
    "delete_any_note":    "supervisor",

    # Alerts
    "ack_alert":          "supervisor",
    "snooze_alert":       "supervisor",
    "resolve_alert":      "supervisor",
    "assign_alert":       "supervisor",
    "ack_own_alert":      "engineer",

    # Models / MLOps
    "view_models":        "supervisor",
    "promote_model":      "admin",
    "rollback_model":     "admin",

    # Users / admin
}


def _rank(role: str | None) -> int:
    """Unknown roles are treated as below viewer."""
    if not role:
        return -1
    return ROLE_RANK.get(role, -1)


def can(capability: str) -> bool:
    """True if the currently authenticated user may exercise `capability`."""
    if not current_user.is_authenticated:
        return False
    required = CAPABILITIES.get(capability)
    if required is None:
        # Unknown capability -> deny and log, so a typo doesn't silently grant access.
        current_app.logger.warning("Unknown capability checked: %s", capability)
        return False
    return _rank(getattr(current_user, "role", None)) >= ROLE_RANK[required]


def require(capability: str):
    """Decorator: abort(403) unless the current user can exercise `capability`."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not can(capability):
                current_app.logger.info(
                    "RBAC deny: user=%s role=%s cap=%s",
                    getattr(current_user, "username", "anon"),
                    getattr(current_user, "role", None),
                    capability,
                )
                abort(403)
            return fn(*args, **kwargs)
        return wrapper
    return decorator