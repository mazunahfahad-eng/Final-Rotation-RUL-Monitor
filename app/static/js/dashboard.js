/**
 * Fleet dashboard sparklines.
 *
 * Each .sparkline canvas lazily loads its own /api/history/<id> when it
 * scrolls near the viewport, and a small concurrency-limited queue keeps
 * the browser from firing 200 parallel requests on first paint.
 *
 * Re-draws on `themechange` (dispatched by app.js) and on window resize,
 * using cached trend data so neither triggers a refetch.
 *
 * Canvas is rendered DPR-aware for crisp lines on retina displays.
 */
(function () {
  "use strict";

  var canvases = Array.prototype.slice.call(
    document.querySelectorAll(".sparkline")
  );
  if (!canvases.length) return;

  // engineId -> array of {cycle, rul, ...} | null (fetch failed)
  var cache = Object.create(null);
  var queue = [];
  var inflight = 0;
  var MAX_INFLIGHT = 4;

  // Theme + color helpers
  function cssVar(name, fallback) {
    var v = getComputedStyle(document.documentElement).getPropertyValue(name);
    v = v && v.trim();
    return v || fallback;
  }

  function statusColor(rul) {
    if (rul == null) return cssVar("--text-muted", "#6b7280");
    if (rul <= 20) return cssVar("--critical", "#b3261e");
    if (rul <= 50) return cssVar("--watch",    "#8a5a00");
    return cssVar("--healthy", "#1b6e3a");
  }

  // DPR-aware canvas sizing
  function sizeCanvas(canvas) {
    var dpr = window.devicePixelRatio || 1;
    var w = canvas.clientWidth || parseInt(canvas.getAttribute("width"), 10) || 100;
    var h = canvas.clientHeight || parseInt(canvas.getAttribute("height"), 10) || 30;

    canvas.width  = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    // Freeze the CSS layout size so we don't fight the HTML attributes.
    canvas.style.width  = w + "px";
    canvas.style.height = h + "px";

    var ctx = canvas.getContext("2d");
    // Draw in CSS-pixel coordinates from here on.
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx: ctx, w: w, h: h };
  }

  // Drawing
  function drawPlaceholder(ctx, w, h) {
    var pad = 3;
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = cssVar("--border", "#d9dce1");
    ctx.setLineDash([2, 2]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad, h / 2);
    ctx.lineTo(w - pad, h / 2);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  function draw(canvas, trend) {
    var dims = sizeCanvas(canvas);
    var ctx = dims.ctx, w = dims.w, h = dims.h, pad = 3;

    if (!trend || trend.length < 2) {
      drawPlaceholder(ctx, w, h);
      return;
    }

    var vals = trend.map(function (p) { return p.rul; });
    var min = Math.min.apply(null, vals);
    var max = Math.max.apply(null, vals);
    var range = Math.max(1, max - min);

    ctx.clearRect(0, 0, w, h);

    // Baseline for visual grounding (subtle, not a full axis).
    ctx.strokeStyle = cssVar("--border", "#d9dce1");
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad, h - pad);
    ctx.lineTo(w - pad, h - pad);
    ctx.stroke();

    // Trend line.
    ctx.strokeStyle = cssVar("--accent", "#2b5c8a");
    ctx.lineWidth = 1.5;
    ctx.lineJoin = "round";
    ctx.beginPath();
    trend.forEach(function (p, i) {
      var x = pad + (i / (trend.length - 1)) * (w - 2 * pad);
      var y = h - pad - ((p.rul - min) / range) * (h - 2 * pad);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // End-point marker coloured by current severity — the operator's eye
    // lands on the *latest* value, which is what matters.
    var last = vals[vals.length - 1];
    var lastX = w - pad;
    var lastY = h - pad - ((last - min) / range) * (h - 2 * pad);
    ctx.fillStyle = statusColor(last);
    ctx.beginPath();
    ctx.arc(lastX, lastY, 2.5, 0, Math.PI * 2);
    ctx.fill();
  }

  // Fetch queue (throttled)
  function pump() {
    while (inflight < MAX_INFLIGHT && queue.length) {
      fetchOne(queue.shift());
    }
  }

  function fetchOne(engineId) {
    inflight++;
    fetch("/api/history/" + encodeURIComponent(engineId), {
      headers: { Accept: "application/json" },
    })
      .then(function (r) {
        if (!r.ok) throw new Error("history " + r.status);
        return r.json();
      })
      .then(function (data) {
        cache[engineId] = (data && data.rul_trend) || [];
        renderFor(engineId);
      })
      .catch(function (err) {
        cache[engineId] = null;
        renderFor(engineId);
        if (window.console) console.warn("sparkline failed for", engineId, err);
      })
      .then(function () {
        inflight--;
        pump();
      });
  }

  function renderFor(engineId) {
    canvases.forEach(function (canvas) {
      if (canvas.dataset.engine === String(engineId)) {
        draw(canvas, cache[engineId]);
      }
    });
  }

  function request(engineId) {
    if (engineId in cache) {
      renderFor(engineId);   // already have it — just redraw
      return;
    }
    queue.push(engineId);
    pump();
  }

  // Lazy-load sparklines as they scroll into view
  if ("IntersectionObserver" in window) {
    var io = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (!entry.isIntersecting) return;
          io.unobserve(entry.target);
          var id = entry.target.dataset.engine;
          if (id) request(id);
        });
      },
      { rootMargin: "150px 0px", threshold: 0.01 }
    );
    canvases.forEach(function (c) { io.observe(c); });
  } else {
    // No IO support: fall back to queued full load (still throttled).
    canvases.forEach(function (c) {
      var id = c.dataset.engine;
      if (id) request(id);
    });
  }

  // ---- Re-draw on theme change (uses cache, no refetch) ----------------
  window.addEventListener("themechange", function () {
    canvases.forEach(function (canvas) {
      var id = canvas.dataset.engine;
      if (id in cache) draw(canvas, cache[id]);
    });
  });

  // ---- Re-draw on resize (DPR/size can change when moving monitors) ----
  var resizeTimer = null;
  window.addEventListener("resize", function () {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () {
      canvases.forEach(function (canvas) {
        var id = canvas.dataset.engine;
        if (id in cache) draw(canvas, cache[id]);
      });
    }, 150);
  });
})();