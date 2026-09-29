"""
Fleet streamer — simulates live telemetry so the UI feels like a real plant
instead of a CSV dump.

How it works

Each `Engine` carries two things:
  1. `SensorReading` rows for cycles already "delivered" to the app.
  2. `trajectory_buffer` — a JSON list of {cycle, op_setting_*, sensors{}} for
     cycles still waiting to be revealed.

On every tick the streamer advances a subset of engines by `STREAM_BATCH`
cycles, appending one `SensorReading` per revealed cycle. Every
`STREAM_PREDICT_EVERY` cycles (per engine) it re-runs inference on the last
`model.seq_len` readings and writes a new `Prediction`. If the new RUL is
below the critical threshold and there isn't already an open alert, it opens
one.

Design choices

* Thread-based (no APScheduler dep). One worker thread + a lock is plenty.
* Each engine has its own `next_tick` offset so engines drift apart in time,
  which is what a real fleet looks like.
* We commit per tick in small batches — a slow tick never blocks the UI.
* The streamer is idempotent: if it restarts it continues from the current
  cursor and buffer, so no data is lost or duplicated.
* Backpressure: we never let the *unrevealed* buffer exceed
  `MAX_LEAD_CYCLES`. This is a safety valve for tests / long-running demos.

Multi-worker safety

Under gunicorn with >1 worker, only ONE process may run the streamer, or
readings collide on the (engine_id, cycle) unique constraint and predictions
multiply. `should_run_streamer()` below reads `STREAMER_LEADER`:

    STREAMER_LEADER=1  -> this worker runs the streamer
    STREAMER_LEADER unset/other -> this worker does not

For a single-process dev server (`python run.py`) the env var defaults to
running, so nothing changes locally. For gunicorn, see the post_fork hook
documented at the top of run.py.
"""
from __future__ import annotations

import os
import random
import socket
import threading
import time
from datetime import datetime

import numpy as np

from app import db
from app.inference import get_model
from app.models import Alert, Engine, Prediction, SensorReading
from app.predictor import predict_for_window


# Config (env-driven so ops can tune without a redeploy)

TICK_SECONDS    = int(os.environ.get("STREAM_TICK", "8"))
BATCH_SIZE      = int(os.environ.get("STREAM_BATCH", "1"))
PREDICT_EVERY   = int(os.environ.get("STREAM_PREDICT_EVERY", "5"))
# Jitter: each engine gets an independent next-tick delay in this range (sec).
JITTER_MIN      = float(os.environ.get("STREAM_JITTER_MIN", "0.5"))
JITTER_MAX      = float(os.environ.get("STREAM_JITTER_MAX", "2.5"))
# Hard cap on how far we let un-delivered history pile up.
MAX_LEAD_CYCLES = int(os.environ.get("STREAM_MAX_LEAD", "4000"))
# How many consecutive failed ticks before we give up and stop the thread.
MAX_CONSECUTIVE_FAILURES = int(os.environ.get("STREAM_MAX_FAILURES", "10"))


def should_run_streamer() -> bool:
    """
    Decide whether *this process* should own the streamer.

    Rules:
      * TESTING or ENABLE_STREAMER=0 -> never (handled by create_app, mirrored
        here so the streamer is defensive if called directly).
      * STREAMER_LEADER unset      -> run (single-process dev default).
      * STREAMER_LEADER="1"        -> run.
      * STREAMER_LEADER="0"/other  -> do not run.

    Under gunicorn, set STREAMER_LEADER=1 on exactly one worker (see the
    post_fork hook documented in run.py). The Flask app calls this before
    instantiating FleetStreamer; FleetStreamer.start() also re-checks to be
    safe if anyone instantiates it manually.
    """
    val = os.environ.get("STREAMER_LEADER")
    if val is None:
        return True
    return val.strip() == "1"


class FleetStreamer:
    def __init__(self, app):
        self.app = app
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # engine_id -> epoch seconds at which this engine is next due
        self._next_due: dict[int, float] = {}
        # protects _next_due / the tick loop from double-start
        self._lock = threading.Lock()
        self._started = False
        # small counters useful for /readyz
        self.ticks = 0
        self.reveals = 0
        self.predictions = 0
        self.failures = 0
        # Identity of the process that owns this streamer, for /readyz.
        self.owner_pid = os.getpid()
        self.owner_host = _hostname()


    # Lifecycle

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            if not should_run_streamer():
                self.app.logger.info(
                    "streamer disabled on this worker (STREAMER_LEADER=%r, pid=%d)",
                    os.environ.get("STREAMER_LEADER"),
                    os.getpid(),
                )
                return
            self._started = True
            self._thread = threading.Thread(
                target=self._run, name="fleet-streamer", daemon=True
            )
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)


    # Main loop

    def _run(self) -> None:
        with self.app.app_context():
            self._bootstrap_schedule()
            while not self._stop.is_set():
                try:
                    self._tick()
                    # A clean tick resets the failure counter — we only give
                    # up on *consecutive* failures, not lifetime ones.
                    self.failures = 0
                except Exception:
                    self.failures += 1
                    # Roll the session back so the next tick can use it.
                    # Without this, a single SQL error poisons the session
                    # and every subsequent tick raises PendingRollbackError.
                    try:
                        db.session.rollback()
                    except Exception:
                        # If even rollback fails, we're in a bad state — log
                        # and let the failure counter decide whether to bail.
                        self.app.logger.exception(
                            "streamer: rollback after failed tick also failed"
                        )

                    self.app.logger.exception(
                        "streamer tick failed (consecutive=%d/%d)",
                        self.failures, MAX_CONSECUTIVE_FAILURES,
                    )

                    if self.failures >= MAX_CONSECUTIVE_FAILURES:
                        self.app.logger.error(
                            "streamer giving up after %d consecutive failures; "
                            "thread will exit. Investigate and restart the app.",
                            self.failures,
                        )
                        break

                # Sleep in small slices so stop() is responsive.
                self._interruptible_sleep(TICK_SECONDS)

            self.app.logger.info(
                "streamer loop exited (pid=%d, ticks=%d, reveals=%d, predictions=%d)",
                os.getpid(), self.ticks, self.reveals, self.predictions,
            )

    def _interruptible_sleep(self, seconds: float) -> None:
        end = time.time() + seconds
        while not self._stop.is_set() and time.time() < end:
            time.sleep(min(0.5, end - time.time()))


    # Scheduling

    def _bootstrap_schedule(self) -> None:
        """Assign each engine a random first-due time so they drift apart."""
        engines = Engine.query.all()
        now = time.time()
        for eng in engines:
            self._next_due.setdefault(eng.id, now + random.uniform(0.0, TICK_SECONDS))
        self.app.logger.info(
            "streamer bootstrapped for %d engines (pid=%d, host=%s)",
            len(engines), self.owner_pid, self.owner_host,
        )

    # One tick

    def _tick(self) -> None:
        self.ticks += 1
        now = time.time()
        due_ids = [eid for eid, t in self._next_due.items() if t <= now]
        if not due_ids:
            return

        engines = Engine.query.filter(Engine.id.in_(due_ids)).all()
        touched_predictions = 0
        for eng in engines:
            revealed = self._reveal(eng, BATCH_SIZE)
            if revealed:
                # Re-run inference if this engine's cycle is aligned with the cadence
                if eng.last_cycle % PREDICT_EVERY == 0:
                    if self._predict(eng):
                        touched_predictions += 1
            # schedule the next due time with a fresh jitter
            self._next_due[eng.id] = now + TICK_SECONDS + random.uniform(JITTER_MIN, JITTER_MAX)

        db.session.commit()
        if touched_predictions:
            self.app.logger.info(
                "tick %d: revealed=%d engines, new_predictions=%d",
                self.ticks, len(engines), touched_predictions,
            )

    # Reveal one or more future cycles from the buffer

    def _reveal(self, eng: Engine, n: int) -> int:
        buffer = eng.trajectory_buffer or []
        if not buffer:
            return 0

        # Buffer is stored oldest-first; we consume from the front.
        # Backpressure: if the tail is enormous, trim to MAX_LEAD_CYCLES.
        if len(buffer) > MAX_LEAD_CYCLES:
            buffer = buffer[-MAX_LEAD_CYCLES:]

        to_take = buffer[:n]
        if not to_take:
            return 0

        for entry in to_take:
            reading = SensorReading(
                engine_id=eng.id,
                cycle=int(entry["cycle"]),
                op_setting_1=entry.get("op_setting_1"),
                op_setting_2=entry.get("op_setting_2"),
                op_setting_3=entry.get("op_setting_3"),
                sensors=entry.get("sensors", {}),
            )
            db.session.add(reading)
            self.reveals += 1

        eng.last_cycle = int(to_take[-1]["cycle"])
        eng.last_reading_at = datetime.utcnow()
        eng.trajectory_buffer = buffer[n:]
        return len(to_take)

    # Inference on the current window

    def _predict(self, eng: Engine) -> bool:
        try:
            model = get_model(eng.dataset)
        except Exception as exc:
            self.app.logger.warning("model load failed for %s: %s", eng.dataset, exc)
            return False

        # Pull the last seq_len readings in cycle order.
        readings = (
            SensorReading.query
            .filter_by(engine_id=eng.id)
            .order_by(SensorReading.cycle.desc())
            .limit(model.seq_len)
            .all()
        )
        if len(readings) < model.seq_len:
            return False
        readings.reverse()

        # Build the (seq_len, n_features) matrix in the model's column order.
        mat = np.zeros((model.seq_len, len(model.feature_cols)), dtype=np.float32)
        for t, reading in enumerate(readings):
            for j, col in enumerate(model.feature_cols):
                mat[t, j] = reading.value(col)

        pred = predict_for_window(eng.dataset, mat)

        # The model was resolved from the active ModelVersion for this exact
        # prediction. Persist that version so the prediction is traceable.
        mv_id = model.model_version_id

        prediction = Prediction(
            engine_id=eng.id,
            model_version_id=mv_id,
            rul=pred["rul"],
            ci_low=pred["ci_low"],
            ci_high=pred["ci_high"],
            confidence_label=pred["confidence_label"],
            input_cycle=eng.last_cycle,
            top_sensors=pred["top_sensors"],
        )
        db.session.add(prediction)
        db.session.flush()
        self.predictions += 1

        # Alerting: open one if we cross the critical threshold and none is open.
        if pred["rul"] <= 20:
            already_open = (
                Alert.query
                .filter_by(engine_id=eng.id)
                .filter(Alert.status.in_(["open", "acknowledged"]))
                .first()
            )
            if not already_open:
                db.session.add(
                    Alert(engine_id=eng.id, prediction_id=prediction.id, status="open")
                )

        return True


# Small helper for /readyz to identify which worker owns the streamer.

def _hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return "unknown"