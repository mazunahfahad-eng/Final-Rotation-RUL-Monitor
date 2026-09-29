/**
 * Global UI behaviour.
 *
 * Three independent concerns, each in its own IIFE:
 *   1. Theme management — persisted choice, OS fallback, `themechange` event.
 *   2. Keyboard shortcuts — "/" focuses search, Escape blurs it.
 *   3. Stream-status poller — drives #stream-chip and the dashboard KPI tile.
 *
 * Contracts other scripts rely on:
 *   • `window.themechange` CustomEvent (detail: { theme }) on every flip.
 *     engine.js listens for this to re-color its charts.
 *   • `window.__rulBootstrapTheme()` — callable from an inline <head> script
 *     so the correct theme is applied before first paint (no FOUC).
 */
(function () {
  "use strict";

  var STORAGE_KEY = "theme";
  var root = document.documentElement;

  // Safe localStorage
  var store = {
    get: function (k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set: function (k, v) { try { localStorage.setItem(k, v); } catch (e) {} },
  };

  // Theme bootstrapper (also usable from <head>)
  window.__rulBootstrapTheme = function () {
    try {
      var stored = store.get(STORAGE_KEY);
      var theme = stored
        ? stored
        : (window.matchMedia &&
           window.matchMedia("(prefers-color-scheme: light)").matches
            ? "light"
            : "dark");
      root.setAttribute("data-theme", theme);
      return theme;
    } catch (e) {
      return root.getAttribute("data-theme") || "dark";
    }
  };

  // ---- Apply on load (idempotent if <head> already did it) -------------
  window.__rulBootstrapTheme();

  function setTheme(next, opts) {
    if (next !== "light" && next !== "dark") return;
    if (next === root.getAttribute("data-theme")) return;
    root.setAttribute("data-theme", next);
    store.set(STORAGE_KEY, next);
    if (!opts || opts.broadcast !== false) {
      window.dispatchEvent(new CustomEvent("themechange", {
        detail: { theme: next },
      }));
    }
  }

  // Toggle button
  var toggle = document.getElementById("theme-toggle");
  if (toggle) {
    toggle.addEventListener("click", function () {
      setTheme(root.getAttribute("data-theme") === "dark" ? "light" : "dark");
    });
  }

  // Follow OS theme changes *only if user hasn't chosen
  if (window.matchMedia) {
    var mq = window.matchMedia("(prefers-color-scheme: light)");
    var onSystemChange = function (e) {
      if (store.get(STORAGE_KEY)) return; // user has an explicit choice
      setTheme(e.matches ? "light" : "dark");
    };
    if (mq.addEventListener) mq.addEventListener("change", onSystemChange);
    else if (mq.addListener) mq.addListener(onSystemChange);
  }

  // Keyboard shortcuts
  function isTypingTarget(el) {
    if (!el) return false;
    var tag = (el.tagName || "").toUpperCase();
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return true;
    return el.isContentEditable === true;
  }

  document.addEventListener("keydown", function (e) {
    // Bail on any modifier — Ctrl+/, Cmd+/, Alt+/ are browser shortcuts.
    if (e.ctrlKey || e.metaKey || e.altKey) return;

    if (e.key === "/") {
      if (isTypingTarget(document.activeElement)) return;
      var search = document.querySelector('input[name="q"]');
      if (search) {
        e.preventDefault();
        search.focus();
        if (typeof search.select === "function") search.select();
      }
      return;
    }

    // Escape clears focus from the search box (common expectation).
    if (e.key === "Escape" && document.activeElement) {
      var tag = (document.activeElement.tagName || "").toUpperCase();
      if (tag === "INPUT" && document.activeElement.name === "q") {
        document.activeElement.blur();
      }
    }
  });
})();


/**
 * Stream-status poller.
 *
 * Pings /api/summary on a per-page cadence and drives the two status
 * indicators in the UI:
 *   • #stream-chip           — topbar chip (every page)
 *   • #stream-dot / #stream-state / #stream-detail / #last-update
 *                            — dashboard KPI tile (dashboard only)
 *
 * States map to the CSS selectors already defined in style.css:
 *   connecting | live | stale | down
 *
 * Staleness is detected by watching the streamer's tick counter. If it
 * hasn't advanced in STALE_AFTER_MS, the chip flips to "stale" — which is
 * exactly the failure mode the streamer rollback fix addresses (thread
 * alive but doing nothing).
 */
(function initStreamStatus() {
  "use strict";

  var topChip    = document.getElementById("stream-chip");
  var dashDot    = document.getElementById("stream-dot");
  var dashState  = document.getElementById("stream-state");
  var dashDetail = document.getElementById("stream-detail");
  var lastUpdEl  = document.getElementById("last-update");

  if (!topChip && !dashDot) return; // no indicators on this page

  // 5s on the dashboard (has the KPI tile), 15s everywhere else.
  var POLL_MS = dashDot ? 5000 : 15000;
  // Consider "stale" if the tick counter hasn't moved in 4 poll intervals.
  var STALE_AFTER_MS = POLL_MS * 4;

  var lastTicks = null;
  var lastTickChangeAt = Date.now();
  var timer = null;
  var inflight = false;

  function setState(state, label, detail) {
    if (topChip) {
      var d = topChip.querySelector(".stream-dot");
      var l = topChip.querySelector(".stream-label");
      if (d) d.setAttribute("data-state", state);
      if (l) l.textContent = label;
      topChip.setAttribute("title", detail || label);
    }
    if (dashDot) {
      dashDot.setAttribute("data-state", state);
      if (dashState)  dashState.textContent  = label;
      if (dashDetail) dashDetail.textContent = detail || "";
    }
  }

  function relTime(iso) {
    if (!iso) return "";
    var diff = (Date.now() - new Date(iso).getTime()) / 1000;
    if (isNaN(diff)) return "";
    diff = Math.max(0, diff);
    if (diff < 5)     return "just now";
    if (diff < 60)    return Math.floor(diff) + "s ago";
    if (diff < 3600)  return Math.floor(diff / 60) + "m ago";
    if (diff < 86400) return Math.floor(diff / 3600) + "h ago";
    return Math.floor(diff / 86400) + "d ago";
  }

  function tick() {
    if (inflight) return;
    inflight = true;
    fetch("/api/summary", { headers: { Accept: "application/json" } })
      .then(function (r) {
        if (r.status === 401) throw new Error("auth");
        if (!r.ok) throw new Error("http " + r.status);
        return r.json();
      })
      .then(function (data) {
        var s = data.streamer;
        if (!s) {
          setState("down", "streamer off", "Streamer disabled in this env");
          return;
        }
        if (!s.alive) {
          setState("down", "streamer offline", "Thread not alive");
          return;
        }
        if (lastTicks === null || s.ticks !== lastTicks) {
          lastTicks = s.ticks;
          lastTickChangeAt = Date.now();
        }
        var age = Date.now() - lastTickChangeAt;
        if (age > STALE_AFTER_MS) {
          setState("stale", "Stale",
                   "No new ticks for " + Math.round(age / 1000) + "s");
        } else {
          setState("live", "Live",
                   s.ticks + " ticks · " + s.predictions + " predictions");
        }
        if (lastUpdEl && data.last_update) {
          lastUpdEl.textContent = relTime(data.last_update);
        }
      })
      .catch(function (err) {
        var auth = err && err.message === "auth";
        setState("down",
                 auth ? "signed out" : "no data",
                 auth ? "Session expired — sign in again"
                      : "Couldn't reach /api/summary");
      })
      .then(function () { inflight = false; });
  }

  function start() {
    tick(); // immediate first read
    if (timer) clearInterval(timer);
    timer = setInterval(function () {
      if (document.hidden) return; // don't poll hidden tabs
      tick();
    }, POLL_MS);
  }

  // Catch up the moment the tab regains focus.
  document.addEventListener("visibilitychange", function () {
    if (!document.hidden) tick();
  });

  start();
})();


/**
 * Delegated toggles for controls rendered by server-side templates
 * (alerts table + note cards). Delegated on document because htmx
 * swaps these rows/cards in and out of the DOM.
 *
 *   [data-snooze-toggle="<form id>"]  — Snooze button on an alert row
 *   [data-toggle-edit="<article id>"] — Edit button on a note card
 *   [data-cancel-edit="<article id>"] — Cancel button inside the edit form
 */
(function initInlineToggles() {
  "use strict";

  document.addEventListener("click", function (e) {
    var snoozeBtn = e.target.closest("[data-snooze-toggle]");
    if (snoozeBtn) {
      var form = document.getElementById(snoozeBtn.getAttribute("data-snooze-toggle"));
      if (form) {
        var willShow = form.hasAttribute("hidden");
        form.hidden = !willShow;
        snoozeBtn.setAttribute("aria-expanded", String(willShow));
        if (willShow) {
          var reasonInput = form.querySelector('input[name="reason"]');
          if (reasonInput) reasonInput.focus();
        }
      }
      return;
    }

    var editBtn = e.target.closest("[data-toggle-edit]");
    if (editBtn) {
      var card = document.getElementById(editBtn.getAttribute("data-toggle-edit"));
      if (card) {
        var editForm = card.querySelector(".note-edit-form");
        var view = card.querySelector(".note-view");
        if (editForm) editForm.hidden = !editForm.hidden;
        if (view) view.hidden = !view.hidden;
      }
      return;
    }

    var cancelBtn = e.target.closest("[data-cancel-edit]");
    if (cancelBtn) {
      var cancelCard = document.getElementById(cancelBtn.getAttribute("data-cancel-edit"));
      if (cancelCard) {
        var cancelForm = cancelCard.querySelector(".note-edit-form");
        var cancelView = cancelCard.querySelector(".note-view");
        if (cancelForm) cancelForm.hidden = true;
        if (cancelView) cancelView.hidden = false;
      }
    }
  });
})();