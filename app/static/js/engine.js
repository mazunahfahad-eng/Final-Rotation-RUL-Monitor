/**
 * Engine detail page — charts + note draft autosave.
 *
 * Both charts share one /api/history/<id> fetch. Colors are read from the
 * CSS custom properties at render time and re-applied on `themechange`
 * (dispatched by app.js) so charts follow the light/dark toggle.
 *
 * The RUL chart renders the 80% CI as a filled band and draws the
 * critical/watch thresholds using chartjs-plugin-annotation — matching the
 * legend already printed in engine_detail.html.
 *
 * The sensor chart normalizes each series to its own [0,1] range so
 * small-range sensors (e.g. S15 ≈ 8.3–8.6) remain legible alongside
 * large-range ones (e.g. S3 ≈ 1570–1620). Raw values are preserved in the
 * tooltip; the y-axis reads "normalized (0–1 per sensor)".
 */
(function () {
  "use strict";

  // Note draft autosave
  (function initNoteDraft() {
    var form = document.getElementById("note-form");
    if (!form) return;

    var body = form.querySelector('textarea[name="body"]');
    var indicator = document.getElementById("save-indicator");
    if (!body || !indicator) return;

    var draftKey = "note-draft-" + window.location.pathname;
    var saveTimer = null;
    var storage = {
      get: function (k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
      set: function (k, v) { try { localStorage.setItem(k, v); } catch (e) {} },
      del: function (k) { try { localStorage.removeItem(k); } catch (e) {} },
    };

    var saved = storage.get(draftKey);
    if (saved) body.value = saved;

    body.addEventListener("input", function () {
      clearTimeout(saveTimer);
      saveTimer = setTimeout(function () {
        storage.set(draftKey, body.value);
        indicator.textContent = "Saved just now";
        setTimeout(function () {
          indicator.textContent = "Saved a moment ago";
        }, 4000);
      }, 600);
    });

    form.addEventListener("htmx:afterRequest", function (evt) {
      if (evt.detail.successful) {
        storage.del(draftKey);
        body.value = "";
        indicator.textContent = "";
      } else {
        indicator.textContent =
          "Couldn't save note. Your draft is safe. Try again.";
      }
    });
  })();

  // Chart helpers
  function cssVar(name, fallback) {
    var v = getComputedStyle(document.documentElement).getPropertyValue(name);
    v = v && v.trim();
    return v || fallback;
  }

  function themeColors() {
    return {
      text:     cssVar("--text-muted", "#9aa1ab"),
      grid:     cssVar("--border",     "#2c313a"),
      accent:   cssVar("--accent",     "#6fa8d6"),
      watch:    cssVar("--watch",      "#e0a83a"),
      critical: cssVar("--critical",   "#ff6b60"),
      healthy:  cssVar("--healthy",    "#4fbf74"),
    };
  }

  function baseScales(c, yTitle) {
    return {
      x: {
        title: { display: true, text: "cycle", color: c.text },
        ticks: { color: c.text, maxTicksLimit: 10 },
        grid:  { color: c.grid },
      },
      y: {
        title: yTitle ? { display: true, text: yTitle, color: c.text } : undefined,
        ticks: { color: c.text },
        grid:  { color: c.grid },
      },
    };
  }

  // Shared data fetch
  var trendCanvas  = document.getElementById("rul-trend-chart");
  var sensorCanvas = document.getElementById("sensor-chart");
  var engineId =
    (trendCanvas  && trendCanvas.dataset.engine) ||
    (sensorCanvas && sensorCanvas.dataset.engine);
  if (!engineId) return;

  var charts = [];

  fetch("/api/history/" + encodeURIComponent(engineId), {
    headers: { Accept: "application/json" },
  })
    .then(function (r) {
      if (!r.ok) throw new Error("history " + r.status);
      return r.json();
    })
    .then(function (data) {
      if (trendCanvas  && window.Chart) renderTrend(trendCanvas, data);
      if (sensorCanvas && window.Chart) renderSensors(sensorCanvas, data);
      window.addEventListener("themechange", restyle);
    })
    .catch(function (err) {
      [trendCanvas, sensorCanvas].forEach(function (cv) {
        if (!cv) return;
        var msg = document.createElement("p");
        msg.className = "empty-state";
        msg.textContent = "Couldn't load chart data. Refresh to retry.";
        cv.replaceWith(msg);
      });
      if (window.console) console.warn("engine chart load failed:", err);
    });

  // RUL trend: point line + CI band + thresholds
  function renderTrend(canvas, data) {
    var c = themeColors();
    var trend = data.rul_trend || [];
    var labels = trend.map(function (p) { return p.cycle; });

    var datasets = [
      {
        label: "CI high",
        data: trend.map(function (p) { return p.ci_high; }),
        borderColor: "transparent",
        backgroundColor: "transparent",
        pointRadius: 0,
        fill: "+1",
      },
      {
        label: "CI low",
        data: trend.map(function (p) { return p.ci_low; }),
        borderColor: "transparent",
        backgroundColor: "color-mix(in srgb, " + c.watch + " 20%, transparent)",
        pointRadius: 0,
        fill: "-1",
      },
      {
        label: "Predicted RUL",
        data: trend.map(function (p) { return p.rul; }),
        borderColor: c.watch,
        backgroundColor: "transparent",
        borderWidth: 1.5,
        pointRadius: 2,
        tension: 0.1,
      },
    ];

    var t = data.thresholds || {};
    var annotations = {};
    if (t.critical != null) {
      annotations.critical = {
        type: "line",
        yMin: t.critical, yMax: t.critical,
        borderColor: c.critical, borderWidth: 1, borderDash: [4, 4],
        label: {
          display: true, content: "critical (" + t.critical + ")",
          color: c.critical, position: "end", backgroundColor: "transparent",
        },
      };
    }
    if (t.watch != null) {
      annotations.watch = {
        type: "line",
        yMin: t.watch, yMax: t.watch,
        borderColor: c.accent, borderWidth: 1, borderDash: [4, 4],
        label: {
          display: true, content: "watch (" + t.watch + ")",
          color: c.accent, position: "end", backgroundColor: "transparent",
        },
      };
    }

    var chart = new Chart(canvas, {
      type: "line",
      data: { labels: labels, datasets: datasets },
      options: {
        responsive: true,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: {
            labels: {
              color: c.text,
              filter: function (item) { return item.text === "Predicted RUL"; },
            },
          },
          tooltip: { mode: "index", intersect: false },
          annotation: { annotations: annotations },
        },
        scales: baseScales(c, "RUL (cycles)"),
      },
    });
    charts.push({ chart: chart, kind: "trend" });
  }

  // Sensor series (normalized per-sensor)
  // Each series is drawn on its own [0,1] range so a sensor whose raw values
  // span 0.3 units (S15) is as readable as one that spans 50 units (S3).
  // Raw values are preserved for the tooltip; the y-axis is relabeled so no
  // one mistakes the normalized line for the sensor's native units.
  function renderSensors(canvas, data) {
    var c = themeColors();
    var palette = [c.accent, c.watch, c.healthy, c.critical];
    var names = Object.keys(data.series || {});

    // Per-series raw min/max, kept aside for tooltip formatting.
    var ranges = {};
    names.forEach(function (name) {
      var vals = (data.series[name] || []).filter(function (v) {
        return v != null && isFinite(v);
      });
      if (!vals.length) { ranges[name] = null; return; }
      var lo = Math.min.apply(null, vals);
      var hi = Math.max.apply(null, vals);
      // Perfectly flat series: pad the range so it renders at y=0.5 rather
      // than dividing by zero and producing NaNs.
      if (hi - lo < 1e-9) { lo -= 0.5; hi += 0.5; }
      ranges[name] = { lo: lo, hi: hi };
    });

    var datasets = names.map(function (name, i) {
      var r = ranges[name];
      var raw = data.series[name] || [];
      var normalized = r
        ? raw.map(function (v) {
            return (v == null || !isFinite(v))
              ? null
              : (v - r.lo) / (r.hi - r.lo);
          })
        : raw;

      return {
        label: name.replace("sensor_measure_", "S"),
        data: normalized,
        // Preserved for the tooltip callback below; not sent to Chart.js.
        _raw: raw,
        _range: r,
        borderColor: palette[i % palette.length],
        backgroundColor: "transparent",
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.15,
      };
    });

    var chart = new Chart(canvas, {
      type: "line",
      data: { labels: data.cycles, datasets: datasets },
      options: {
        responsive: true,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { labels: { color: c.text } },
          tooltip: {
            callbacks: {
              // Display the raw sensor value, not the 0–1 normalized one.
              label: function (ctx) {
                var ds = ctx.dataset;
                var raw = ds._raw ? ds._raw[ctx.dataIndex] : ctx.parsed.y;
                if (raw == null || !isFinite(raw)) return ds.label + ": —";
                var r = ds._range;
                var rangeTxt = r
                  ? "  [" + r.lo.toFixed(2) + "–" + r.hi.toFixed(2) + "]"
                  : "";
                return ds.label + ": " + Number(raw).toFixed(2) + rangeTxt;
              },
            },
          },
        },
        scales: {
          x: {
            title: { display: true, text: "cycle", color: c.text },
            ticks: { color: c.text, maxTicksLimit: 10 },
            grid:  { color: c.grid },
          },
          y: {
            title: {
              display: true,
              text: "normalized (0–1 per sensor)",
              color: c.text,
            },
            min: 0,
            max: 1,
            ticks: { color: c.text },
            grid:  { color: c.grid },
          },
        },
      },
    });
    charts.push({ chart: chart, kind: "sensors", palette: palette });
  }

  // Re-theme on toggle 
  function restyle() {
    var c = themeColors();
    charts.forEach(function (entry) {
      var ch = entry.chart;
      var opts = ch.options;

      ["x", "y"].forEach(function (axis) {
        var a = opts.scales[axis];
        if (!a) return;
        if (a.ticks) a.ticks.color = c.text;
        if (a.grid)  a.grid.color  = c.grid;
        if (a.title) a.title.color = c.text;
      });
      if (opts.plugins.legend && opts.plugins.legend.labels) {
        opts.plugins.legend.labels.color = c.text;
      }

      if (entry.kind === "trend") {
        ch.data.datasets[2].borderColor = c.watch;
        ch.data.datasets[1].backgroundColor =
          "color-mix(in srgb, " + c.watch + " 20%, transparent)";
        var ann = opts.plugins.annotation && opts.plugins.annotation.annotations;
        if (ann) {
          if (ann.critical) {
            ann.critical.borderColor = c.critical;
            if (ann.critical.label) ann.critical.label.color = c.critical;
          }
          if (ann.watch) {
            ann.watch.borderColor = c.accent;
            if (ann.watch.label) ann.watch.label.color = c.accent;
          }
        }
      } else if (entry.kind === "sensors") {
        ch.data.datasets.forEach(function (ds, i) {
          ds.borderColor = entry.palette[i % entry.palette.length];
        });
      }
      ch.update("none");
    });
  }
})();