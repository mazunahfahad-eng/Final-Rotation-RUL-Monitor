"""
WSGI entry point.

Used by:
  - gunicorn:  gunicorn -c gunicorn.conf.py run:app
  - local dev: python run.py

The streamer is started inside create_app() (see app/__init__.py). Under
gunicorn with multiple workers, set STREAMER_LEADER=1 on exactly one worker
(see gunicorn.conf.py post_fork hook) so we don't run N concurrent streamers
against the same DB.

Set FLASK_ENV=development to enable the reloader; default is production.
"""
from __future__ import annotations

import os

from app import create_app

# Build the app at import time so `gunicorn run:app` finds it.
app = create_app()


if __name__ == "__main__":
    env = os.environ.get("FLASK_ENV", "production").lower()
    debug = env == "development"
    port = int(os.environ.get("PORT", "5000"))
    host = os.environ.get("HOST", "0.0.0.0")

    # Reloader only in dev; the streamer thread + reloader do not mix well.
    app.run(
        host=host,
        port=port,
        debug=debug,
        use_reloader=debug,
        threaded=True,
    )