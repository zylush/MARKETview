(() => {
  "use strict";

  const API = Object.freeze({
    latest: (symbol) => `/api/v1/eod/latest/${encodeURIComponent(symbol)}`,
    history: (symbol) => `/api/v1/eod/history/${encodeURIComponent(symbol)}`,
    usage: "/api/v1/usage",
  });
  const SYMBOL_PATTERN = /^[A-Za-z0-9][A-Za-z0-9.-]{0,31}$/;
  const $ = (id) => document.getElementById(id);
  const elements = Object.freeze({
    searchForm: $("ticker-search-form"),
    searchInput: $("ticker-search"),
    searchButton: document.querySelector("[data-testid='search-submit']"),
    searchError: $("search-error"),
    historyForm: $("history-range-form"),
    dateFrom: $("date-from"),
    dateTo: $("date-to"),
    logoutForm: $("logout-form"),
  });
  let state = Object.freeze({
    symbol: String(document.body.dataset.initialSymbol || "AAPL").toUpperCase(),
    chart: null,
  });

  function updateState(changes) {
    state = Object.freeze({ ...state, ...changes });
  }

  function setVisible(element, visible) {
    if (element) element.hidden = !visible;
  }

  function setText(id, value) {
    const element = $(id);
    if (element) element.textContent = value ?? "—";
  }

  function displayGlobal(message, isError = false) {
    const element = $("global-status");
    element.textContent = message;
    element.classList.toggle("is-error", isError);
    setVisible(element, Boolean(message));
  }

  function setSectionState(name, status, message = "") {
    ["loading", "error", "empty", "content"].forEach((suffix) => {
      const element = $(`${name}-${suffix}`);
      if (element) setVisible(element, suffix === status);
    });
    if (status === "error") setText(`${name}-error`, message);
    if (name === "quote") setVisible($("quote-metrics"), status === "ready");
    if (name === "quote") setVisible($("quote-loading"), status === "loading");
    if (name === "quote") setVisible($("quote-error"), status === "error");
    if (name === "quote") setVisible($("quote-empty"), status === "empty");
    if (status === "ready") setVisible($(`${name}-content`), true);
  }

  function asList(value) {
    if (Array.isArray(value)) return value;
    if (value && Array.isArray(value.items)) return value.items;
    return [];
  }

  function firstDefined(object, keys, fallback = null) {
    for (const key of keys) {
      if (object && object[key] !== undefined && object[key] !== null) return object[key];
    }
    return fallback;
  }

  function formatNumber(value, options = {}) {
    const numeric = Number(value);
    return Number.isFinite(numeric)
      ? new Intl.NumberFormat("en-US", options).format(numeric)
      : "—";
  }

  function formatMoney(value) {
    const numeric = Number(value);
    return Number.isFinite(numeric)
      ? new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 4 }).format(numeric)
      : "—";
  }

  function dateValue(value) {
    if (!value) return "—";
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return String(value);
    return new Intl.DateTimeFormat("en-US", { year: "numeric", month: "short", day: "numeric", timeZone: "UTC" }).format(parsed);
  }

  function dateTimeValue(value) {
    if (!value) return "Reset time unavailable";
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return "Reset time unavailable";
    return `Resets ${new Intl.DateTimeFormat("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
      timeZone: "UTC",
      timeZoneName: "short",
    }).format(parsed)}`;
  }

  function isoDate(date) {
    return date.toISOString().slice(0, 10);
  }

  function setFreshness(meta = {}) {
    const source = meta.source === "cache" ? "Cached" : "Provider";
    const stale = meta.stale ? " · stale fallback" : "";
    setText("data-freshness", `${source}${stale}${meta.as_of ? ` · ${dateValue(meta.as_of)}` : ""}`);
  }

  async function request(url) {
    const response = await fetch(url, {
      method: "GET",
      credentials: "same-origin",
      headers: { Accept: "application/json" },
    });
    let payload = null;
    try {
      payload = await response.json();
    } catch (_error) {
      throw new Error("The server returned an unreadable response.");
    }
    if (!response.ok || payload.success === false) {
      throw new Error(payload?.error?.message || `Request failed (${response.status}).`);
    }
    return payload;
  }

  function normalizeSymbol(value) {
    const normalized = String(value || "").trim().toUpperCase();
    return SYMBOL_PATTERN.test(normalized) ? normalized : null;
  }

  function showSymbolError(message) {
    elements.searchError.textContent = message;
    setVisible(elements.searchError, true);
    elements.searchInput.setAttribute("aria-invalid", "true");
    elements.searchInput.focus();
  }

  function clearSymbolError() {
    elements.searchError.textContent = "";
    setVisible(elements.searchError, false);
    elements.searchInput.removeAttribute("aria-invalid");
  }

  function renderQuote(record, meta = {}) {
    const close = Number(firstDefined(record, ["close", "price"]));
    const open = Number(firstDefined(record, ["open"]));
    const change = close - open;
    const percent = open ? (change / open) * 100 : Number.NaN;
    setText("metric-close", formatMoney(close));
    setText("metric-open", formatMoney(open));
    setText("metric-high", formatMoney(firstDefined(record, ["high"])));
    setText("metric-low", formatMoney(firstDefined(record, ["low"])));
    setText("metric-volume", formatNumber(firstDefined(record, ["volume"]), { notation: "compact", maximumFractionDigits: 2 }));
    setText("quote-date", `As of ${dateValue(firstDefined(record, ["date", "timestamp"], meta.as_of))}`);
    const badge = $("metric-change");
    badge.classList.remove("is-positive", "is-negative");
    if (Number.isFinite(change)) {
      const sign = change > 0 ? "+" : "";
      badge.textContent = `${sign}${formatNumber(change, { maximumFractionDigits: 4 })}${Number.isFinite(percent) ? ` (${sign}${formatNumber(percent, { maximumFractionDigits: 2 })}%)` : ""}`;
      badge.classList.add(change >= 0 ? "is-positive" : "is-negative");
    } else {
      badge.textContent = "Change unavailable";
    }
    setSectionState("quote", "ready");
    setFreshness(meta);
  }

  function normalizeHistory(items) {
    return items
      .map((item) => Object.freeze({
        date: firstDefined(item, ["date", "timestamp"]),
        open: firstDefined(item, ["open"]),
        high: firstDefined(item, ["high"]),
        low: firstDefined(item, ["low"]),
        close: firstDefined(item, ["close", "price"]),
        volume: firstDefined(item, ["volume"]),
      }))
      .filter((item) => item.date && Number.isFinite(Number(item.close)))
      .sort((left, right) => new Date(left.date) - new Date(right.date));
  }

  function renderHistoryTable(records) {
    const rows = records.map((record) => {
      const row = document.createElement("tr");
      [dateValue(record.date), formatMoney(record.open), formatMoney(record.high), formatMoney(record.low), formatMoney(record.close), formatNumber(record.volume, { notation: "compact", maximumFractionDigits: 2 })]
        .forEach((value) => {
          const cell = document.createElement("td");
          cell.textContent = value;
          row.append(cell);
        });
      return row;
    });
    $("history-table-body").replaceChildren(...rows);
  }

  function renderChart(records) {
    if (state.chart) state.chart.destroy();
    const canvas = $("price-chart");
    if (!window.Chart) {
      $("history-table-details").open = true;
      canvas.hidden = true;
      return;
    }
    canvas.hidden = false;
    const chart = new window.Chart(canvas.getContext("2d"), {
      type: "line",
      data: {
        labels: records.map((record) => dateValue(record.date)),
        datasets: [{
          label: `${state.symbol} unadjusted close`,
          data: records.map((record) => Number(record.close)),
          borderColor: "#059669",
          backgroundColor: "#059669",
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 5,
          tension: 0.12,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? false : { duration: 220 },
        interaction: { mode: "index", intersect: false },
        plugins: { legend: { display: true, align: "start" } },
        scales: {
          x: { title: { display: true, text: "Trading date" }, grid: { display: false } },
          y: { title: { display: true, text: "Price (USD)" } },
        },
      },
    });
    updateState({ chart });
  }

  function renderHistory(items) {
    const records = normalizeHistory(items);
    if (!records.length) return setSectionState("history", "empty");
    const first = records[0];
    const last = records[records.length - 1];
    const change = Number(last.close) - Number(first.close);
    setText("chart-summary", `${state.symbol} closed at ${formatMoney(last.close)} on ${dateValue(last.date)}, ${change >= 0 ? "up" : "down"} ${formatMoney(Math.abs(change))} from ${dateValue(first.date)} across ${records.length} observations.`);
    renderHistoryTable(records);
    renderChart(records);
    setSectionState("history", "ready");
  }

  function renderUsage(data) {
    const usage = data || {};
    const used = Number(firstDefined(usage, ["requests_used", "used"], 0));
    const limit = Number(firstDefined(usage, ["requests_limit", "limit"], 0));
    const remaining = Number(firstDefined(usage, ["requests_remaining", "remaining"]));
    const resolvedUsed = Number.isFinite(used) ? used : Number.isFinite(remaining) && limit ? limit - remaining : 0;
    const percent = limit > 0 ? Math.min(100, Math.max(0, (resolvedUsed / limit) * 100)) : 0;
    setText("usage-used", formatNumber(resolvedUsed));
    setText("usage-limit", limit ? formatNumber(limit) : "unlimited");
    setText("usage-reset", dateTimeValue(firstDefined(usage, ["reset_at"])));
    const progress = $("usage-progress");
    progress.setAttribute("aria-valuenow", String(Math.round(percent)));
    progress.classList.toggle("is-warning", percent >= 75 && percent < 90);
    progress.classList.toggle("is-critical", percent >= 90);
    $("usage-bar").style.width = `${percent}%`;
    setSectionState("usage", "ready");
  }

  function historyUrl(symbol) {
    const params = new URLSearchParams({
      date_from: elements.dateFrom.value,
      date_to: elements.dateTo.value,
      limit: "1000",
    });
    return `${API.history(symbol)}?${params.toString()}`;
  }

  async function loadHistory(symbol) {
    setSectionState("history", "loading");
    try {
      const response = await request(historyUrl(symbol));
      renderHistory(asList(response.data));
    } catch (error) {
      setSectionState("history", "error", `${error.message} Adjust the range or retry.`);
    }
    await refreshUsage();
  }

  async function refreshUsage() {
    setSectionState("usage", "loading");
    try {
      const response = await request(API.usage);
      renderUsage(response.data);
    } catch (error) {
      setSectionState("usage", "error", error.message);
    }
  }

  async function loadDashboard(symbol) {
    updateState({ symbol });
    elements.searchInput.value = symbol;
    setText("header-symbol", symbol);
    setText("company-symbol", symbol);
    setSectionState("quote", "loading");
    setSectionState("history", "loading");
    setSectionState("usage", "loading");
    displayGlobal("");
    const [latest, history] = await Promise.allSettled([
      request(API.latest(symbol)),
      request(historyUrl(symbol)),
    ]);
    if (latest.status === "fulfilled") {
      const record = Array.isArray(latest.value.data) ? latest.value.data[0] : latest.value.data;
      record ? renderQuote(record, latest.value.meta) : setSectionState("quote", "empty");
    } else setSectionState("quote", "error", latest.reason.message);
    if (history.status === "fulfilled") renderHistory(asList(history.value.data));
    else setSectionState("history", "error", history.reason.message);
    await refreshUsage();
  }

  function initializeDates(days = 365) {
    const end = new Date();
    const start = new Date(end);
    start.setUTCDate(start.getUTCDate() - days);
    elements.dateTo.max = isoDate(end);
    elements.dateFrom.max = isoDate(end);
    elements.dateTo.value = isoDate(end);
    elements.dateFrom.value = isoDate(start);
  }

  function submitSymbol() {
    clearSymbolError();
    const symbol = normalizeSymbol(elements.searchInput.value);
    if (!symbol) {
      showSymbolError("Enter 1–32 letters, numbers, periods, or hyphens; begin with a letter or number.");
      return;
    }
    elements.searchInput.value = symbol;
    loadDashboard(symbol);
  }

  function bindEvents() {
    elements.searchForm.addEventListener("submit", (event) => {
      event.preventDefault();
      submitSymbol();
    });
    elements.searchInput.addEventListener("input", clearSymbolError);
    elements.historyForm.addEventListener("submit", (event) => {
      event.preventDefault();
      if (elements.dateFrom.value > elements.dateTo.value) {
        setSectionState("history", "error", "The start date must be before the end date.");
        return elements.dateFrom.focus();
      }
      document.querySelectorAll(".range-button").forEach((button) => {
        button.classList.remove("is-active");
        button.setAttribute("aria-pressed", "false");
      });
      loadHistory(state.symbol);
    });
    document.querySelectorAll(".range-button").forEach((button) => {
      button.addEventListener("click", () => {
        initializeDates(Number(button.dataset.days));
        document.querySelectorAll(".range-button").forEach((candidate) => {
          candidate.classList.toggle("is-active", candidate === button);
          candidate.setAttribute("aria-pressed", candidate === button ? "true" : "false");
        });
        loadHistory(state.symbol);
      });
    });
    elements.logoutForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      const button = elements.logoutForm.querySelector("button");
      button.disabled = true;
      button.textContent = "Logging out…";
      try {
        const body = new URLSearchParams({ csrf_token: document.querySelector("meta[name='csrf-token']")?.content || "" });
        const response = await fetch("/auth/logout", {
          method: "POST",
          credentials: "same-origin",
          headers: { "Content-Type": "application/x-www-form-urlencoded" },
          body,
        });
        if (!response.ok) throw new Error("Logout request failed");
        window.location.assign("/");
      } catch (_error) {
        displayGlobal("Could not log out. Check your connection and try again.", true);
        button.disabled = false;
        button.textContent = "Log out";
      }
    });
  }

  function initialize() {
    initializeDates();
    bindEvents();
    const initial = normalizeSymbol(state.symbol) || "AAPL";
    loadDashboard(initial);
  }

  initialize();
})();
