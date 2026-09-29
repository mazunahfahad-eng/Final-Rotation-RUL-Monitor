# RUL Monitor

A fleet-monitoring platform for turbofan engines that predicts each engine's Remaining Useful Life (RUL) and turns those predictions into a managed maintenance workflow. It is built on NASA's C-MAPSS benchmark (FD001 and FD003), with role-based access so viewers, engineers, supervisors and admins each see and do only what their job requires.

## Purpose

Predictive-maintenance models usually stop at a number. RUL Monitor carries that number through to action: an engine's predicted RUL crosses a threshold, an alert opens, a supervisor assigns it to an engineer, the engineer acknowledges it, investigates, records notes, and the supervisor resolves it. Every step is audited.

## What it does

- **Fleet dashboard.** All 200 engines (100 from FD001, 100 from FD003) with status, predicted RUL, 80% confidence interval and confidence label. Filterable by dataset, status and tag, sorted worst-first.
- **Engine detail.** Sensor history charts, RUL trend with confidence band, anomaly flags and the sensors driving the current prediction.
- **Live simulation.** A background streamer reveals each engine's trajectory one cycle at a time, re-runs inference and opens alerts as engines cross the critical threshold.
- **Alert work queue.** Alerts are sortable by RUL, age, or assignment to the current user. Supervisors assign, snooze (with a mandatory reason), and resolve. Assigned engineers acknowledge their own alerts.
- **Engineer notes.** Tagged, severity-graded notes with status tracking and an action-taken field.
- **Model registry.** Versioned models per dataset with promote and rollback; exactly one version is active per dataset, and every prediction records the version that produced it.
- **Audit trail.** Every mutating action is logged with user, target and detail.

## Roles

| Role       | Can do                                                                                                                                |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Viewer     | View dashboard, engines and alerts                                                                                                    |
| Engineer   | Everything a viewer can, plus add, edit and delete own notes, and acknowledge alerts assigned to them                                 |
| Supervisor | Everything an engineer can, plus acknowledge, snooze, resolve and assign any alert; edit and delete any note; view the model registry |
| Admin      | Everything a supervisor can, plus promote and roll back models and view the audit log                                                 |

Permissions are expressed as capabilities in `app/rbac.py`, and both routes and templates check the same `can()` helper so UI and API cannot drift apart.

## Alert lifecycle

```
open ──assign──> open (assigned) ──engineer ack──> acknowledged ──> resolved
  │                                                     │
  └──────────────── snooze (reason + duration) ─────────┘
```

Reassigning an acknowledged alert returns it to `open`, since the new engineer has not yet accepted it. An alert opens automatically when an engine's predicted RUL falls to 20 cycles or below, and only one active alert exists per engine.

## Models

| Dataset | Architecture                                                    | RMSE  | R²   |
| ------- | --------------------------------------------------------------- | ----- | ---- |
| FD001   | Attention BiLSTM (2 layers, hidden 96, temporal attention head) | 13.60 | 0.88 |
| FD003   | Plain LSTM (2 layers, hidden 96)                                | 12.39 | 0.89 |

Both use a 30-cycle window, a RUL cap of 125, and an asymmetric smooth-L1 loss that penalises over-prediction twice as heavily as under-prediction, since overestimating remaining life is the costlier error. They were selected in comparison notebooks, retrained in JAX on the full training sets, and exported as plain NumPy weights. Serving uses a from-scratch NumPy forward pass, so no deep-learning framework is needed at runtime.

The 80% confidence interval is the prediction ± 1.28 × the model's held-out RMSE. Feature attribution uses the attention weights for FD001 and a recent-versus-window drift measure for FD003. Operating-condition columns are excluded from the driver ranking because they describe flight conditions, not degradation.

## Architecture

```
app/
  __init__.py     app factory, extensions, context processors
  auth.py         login and logout
  rbac.py         roles and capabilities
  routes.py       dashboard, engine, notes, alerts, models, audit, JSON APIs
  models.py       SQLAlchemy schema
  inference.py    NumPy forward pass over trained weights
  predictor.py    confidence interval and driver logic
  streamer.py     live simulation and alert creation
  seed.py         loads FD001/FD003 test sets and runs initial predictions
  templates/      Jinja + htmx views
  static/         CSS and Chart.js front-end
models_bin/       exported weights and metrics per dataset
train/            resumable JAX training script
```

Stack: Flask, SQLAlchemy (SQLite), Flask-Login, htmx, Chart.js, NumPy.

## Data

The fleet is the C-MAPSS test split: in-service engines whose true end-of-life is known from the `RUL_FD00x.txt` files but hidden from the model. Training data is used only for training. Every cycle of every engine is stored, so charts show real history and the RUL trend reflects how predictions actually evolved.

## JSON API

| Endpoint            | Returns                                              |
| ------------------- | ---------------------------------------------------- |
| `/api/summary`      | Fleet status counts and streamer health              |
| `/api/engines`      | Filtered fleet list                                  |
| `/api/history/<id>` | Sensor series, RUL trend with CI, anomalies, drivers |
| `/api/predict/<id>` | Latest prediction payload                            |

## Known limitations

- No CSV or PDF export.
- Snooze has a fixed duration menu rather than a date picker; there are no bulk alert actions.
- Confidence labels use a fixed interval-width cutoff and are not calibrated against observed coverage.
- Single dev server on SQLite with no cache. This is fine for 200 engines but not for FD002/FD004 scale.
- Only FD001 and FD003 are supported; the multi-condition datasets would need per-condition normalisation.
