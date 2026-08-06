(function () {
  "use strict";

  const API = Object.freeze({
    tickers: "/api/v1/tickers",
    exchanges: "/api/v1/exchanges",
    latest: (symbol) => `/api/v1/eod/latest/${encodeURIComponent(symbol)}`,
    history: (symbol) => `/api/v1/eod/history/${encodeURIComponent(symbol)}`,
    splits: (symbol) => `/api/v1/splits/${encodeURIComponent(symbol)}`,
    dividends: (symbol) => `/api/v1/dividends/${encodeURIComponent(symbol)}`,
    usage: "/api/v1/usage",
  });

  const $ = (id) => document.getElementById(id);
  const elements = Object.freeze({
    globalStatus: $("global-status"),
    searchForm: $("ticker-search-form"),
    searchInput: $("ticker-search"),
    searchButton: document.querySelector("[data-testid='search-submit']"),
    searchResults: $("ticker-results"),
    searchError: $("search-error"),
    historyForm: $("history-form"),
    dateFrom: $("date-from"),
    dateTo: $("date-to"),
    logoutForm: $("logout-form"),
  });

  let state = Object.freeze({
    symbol: (document.body.dataset.initialSymbol || "AAPL").trim().toUpperCase(),
    ticker: null,
    exchanges: [],
    chart: null,
    searchResults: [],
    searchIndex: -1,
  });

  function updateState(patch) {
    state = Object.freeze(Object.assign({}, state, patch));
  }

  function nestedData(data) {
    return data && typeof data === "object" && !Array.isArray(data) && "data" in data ? data.data : data;
  }

  function asList(data) {
    const value = nestedData(data);
    if (Array.isArray(value)) return value;
    if (value && Array.isArray(value.results)) return value.results;
    if (value && Array.isArray(value.items)) return value.items;
    return [];
  }

  function firstDefined(source, keys, fallback = null) {
    if (!source || typeof source !== "object") return fallback;
    for (const key of keys) {
      if (source[key] !== undefined && source[key] !== null && source[key] !== "") return source[key];
    }
    return fallback;
  }

  async function request(path, options = {}) {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 15000);
    const csrfToken = document.querySelector("meta[name='csrf-token']")?.content;
    const headers = Object.assign(
      { Accept: "application/json" },
      csrfToken ? { "X-CSRF-Token": csrfToken } : {},
      options.headers || {},
    );
    try {
      const response = await fetch(path, Object.assign({ credentials: "same-origin", signal: controller.signal }, options, { headers }));
      if (response.status === 401) {
        window.location.assign("/");
        throw new Error("Your session expired. Sign in again.");
      }
      let envelope;
      try {
        envelope = await response.json();
      } catch (_error) {
        throw new Error("The server returned an unreadable response. Try again.");
      }
      if (!response.ok || !envelope || envelope.success !== true) {
        throw new Error(envelope?.error?.message || `Request failed (${response.status}). Try again.`);
      }
      return Object.freeze({ data: nestedData(envelope.data), meta: envelope.meta || {} });
    } catch (error) {
      if (error.name === "AbortError") throw new Error("The request timed out. Check your connection and retry.");
      throw error;
    } finally {
      window.clearTimeout(timeout);
    }
  }

  function setVisible(element, visible) {
    if (element) element.hidden = !visible;
  }

  function setText(id, value) {
    const element = $(id);
    if (element) element.textContent = value === null || value === undefined || value === "" ? "—" : String(value);
  }

  function setSectionState(name, mode, message = "") {
    const loading = $(`${name}-loading`);
    const error = $(`${name}-error`);
    const empty = $(`${name}-empty`);
    const content = $(`${name}-content`) || (name === "quote" ? $("quote-metrics") : name === "metadata" ? $("metadata-list") : null);
    setVisible(loading, mode === "loading");
    setVisible(error, mode === "error");
    setVisible(empty, mode === "empty");
    setVisible(content, mode === "ready");
    if (mode === "error" && error) error.textContent = message;
  }

  function formatNumber(value, options = {}) {
    const number = Number(value);
    return Number.isFinite(number) ? new Intl.NumberFormat(undefined, options).format(number) : "—";
  }

  function formatMoney(value, currency = "USD") {
    const number = Number(value);
    if (!Number.isFinite(number)) return "—";
    try {
      return new Intl.NumberFormat(undefined, { style: "currency", currency, maximumFractionDigits: 4 }).format(number);
    } catch (_error) {
      return formatNumber(number, { maximumFractionDigits: 4 });
    }
  }

  function dateValue(value) {
    if (!value) return "—";
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? String(value).slice(0, 10) : new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeZone: "UTC" }).format(date);
  }

  function isoDate(date) {
    return date.toISOString().slice(0, 10);
  }

  function displayGlobal(message, isError = false) {
    elements.globalStatus.textContent = message;
    elements.globalStatus.classList.toggle("is-error", isError);
    setVisible(elements.globalStatus, Boolean(message));
  }

  function setFreshness(meta) {
    const asOf = firstDefined(meta, ["as_of"]);
    const source = firstDefined(meta, ["source"]);
    const flags = [source, asOf ? `updated ${dateValue(asOf)}` : null, meta.cached ? "cached" : null, meta.stale ? "stale" : null].filter(Boolean);
    setText("data-freshness", flags.length ? flags.join(" · ") : "Live market data");
  }

  function resolveExchange(ticker) {
    const exchangeCode = String(firstDefined(ticker, ["stock_exchange", "exchange", "exchange_code", "exchange_acronym", "mic"], ""));
    return state.exchanges.find((exchange) => ["acronym", "mic", "code", "name"].some((key) => String(exchange[key] || "").toLowerCase() === exchangeCode.toLowerCase())) || {};
  }

  function renderMetadata(ticker) {
    const exchange = resolveExchange(ticker);
    const company = firstDefined(ticker, ["name", "company_name", "company", "security_name"], state.symbol);
    const exchangeName = firstDefined(ticker, ["exchange_name", "stock_exchange_name"], firstDefined(exchange, ["name", "acronym"], firstDefined(ticker, ["exchange", "stock_exchange"], "—")));
    setText("company-name", company);
    setText("company-symbol", state.symbol);
    setText("header-symbol", state.symbol);
    setText("meta-company", company);
    setText("meta-symbol", state.symbol);
    setText("meta-exchange", exchangeName);
    setText("meta-mic", firstDefined(ticker, ["mic", "exchange_mic"], firstDefined(exchange, ["mic"], "—")));
    setText("meta-country", firstDefined(ticker, ["country", "country_name"], firstDefined(exchange, ["country", "country_name"], "—")));
    setText("meta-currency", firstDefined(ticker, ["currency", "currency_code"], firstDefined(exchange, ["currency", "currency_code"], "USD")));
    setSectionState("metadata", "ready");
  }

  function normalizeTicker(item) {
    return Object.freeze({
      raw: item,
      symbol: String(firstDefined(item, ["symbol", "ticker", "code"], "")).toUpperCase(),
      name: String(firstDefined(item, ["name", "company_name", "company", "security_name"], "Unnamed security")),
    });
  }

  function renderSearchResults(items) {
    const normalized = items.map(normalizeTicker).filter((item) => item.symbol);
    updateState({ searchResults: normalized, searchIndex: -1 });
    elements.searchResults.replaceChildren();
    if (!normalized.length) {
      const item = document.createElement("li");
      item.className = "section-message";
      item.textContent = "No securities matched your search. Try a ticker symbol.";
      elements.searchResults.append(item);
    } else {
      normalized.forEach((ticker, index) => {
        const item = document.createElement("li");
        const button = document.createElement("button");
        button.type = "button";
        button.id = `ticker-option-${index}`;
        button.dataset.index = String(index);
        button.setAttribute("role", "option");
        button.setAttribute("aria-selected", "false");
        const symbol = document.createElement("span");
        symbol.className = "result-symbol";
        symbol.textContent = ticker.symbol;
        const company = document.createElement("span");
        company.className = "result-company";
        company.textContent = ticker.name;
        button.append(symbol, company);
        button.addEventListener("click", () => selectTicker(ticker));
        item.append(button);
        elements.searchResults.append(item);
      });
    }
    setVisible(elements.searchResults, true);
    elements.searchInput.setAttribute("aria-expanded", "true");
  }

  function closeSearchResults() {
    setVisible(elements.searchResults, false);
    elements.searchInput.setAttribute("aria-expanded", "false");
    elements.searchInput.removeAttribute("aria-activedescendant");
  }

  function moveSearchFocus(direction) {
    if (!state.searchResults.length || elements.searchResults.hidden) return;
    const nextIndex = (state.searchIndex + direction + state.searchResults.length) % state.searchResults.length;
    updateState({ searchIndex: nextIndex });
    const buttons = elements.searchResults.querySelectorAll("button[role='option']");
    buttons.forEach((button, index) => button.setAttribute("aria-selected", index === nextIndex ? "true" : "false"));
    const active = buttons[nextIndex];
    elements.searchInput.setAttribute("aria-activedescendant", active.id);
    active.scrollIntoView({ block: "nearest" });
  }

  async function runSearch(query) {
    const cleaned = query.trim();
    setVisible(elements.searchError, false);
    if (!cleaned) {
      elements.searchError.textContent = "Enter a ticker or company name.";
      setVisible(elements.searchError, true);
      elements.searchInput.focus();
      return;
    }
    elements.searchButton.disabled = true;
    elements.searchButton.textContent = "Searching…";
    try {
      const params = new URLSearchParams({ search: cleaned, limit: "8" });
      const response = await request(`${API.tickers}?${params.toString()}`);
      renderSearchResults(asList(response.data));
    } catch (error) {
      elements.searchError.textContent = error.message;
      setVisible(elements.searchError, true);
    } finally {
      elements.searchButton.disabled = false;
      elements.searchButton.textContent = "Search";
    }
  }

  function selectTicker(ticker) {
    elements.searchInput.value = `${ticker.symbol} — ${ticker.name}`;
    closeSearchResults();
    updateState({ symbol: ticker.symbol, ticker: ticker.raw });
    renderMetadata(ticker.raw);
    loadSecurity(ticker.symbol);
  }

  function renderQuote(record, meta) {
    const currency = firstDefined(state.ticker, ["currency", "currency_code"], "USD");
    const close = Number(firstDefined(record, ["close", "adj_close", "adjusted_close", "price"]));
    const open = Number(firstDefined(record, ["open", "adj_open", "adjusted_open"]));
    const explicitChange = Number(firstDefined(record, ["change", "price_change"]));
    const change = Number.isFinite(explicitChange) ? explicitChange : close - open;
    const explicitPercent = Number(firstDefined(record, ["change_percent", "percent_change", "change_pct"]));
    const percent = Number.isFinite(explicitPercent) ? explicitPercent : open ? (change / open) * 100 : NaN;
    setText("metric-close", formatMoney(close, currency));
    setText("metric-open", formatMoney(open, currency));
    setText("metric-high", formatMoney(firstDefined(record, ["high", "adj_high", "adjusted_high"]), currency));
    setText("metric-low", formatMoney(firstDefined(record, ["low", "adj_low", "adjusted_low"]), currency));
    setText("metric-volume", formatNumber(firstDefined(record, ["volume", "adj_volume", "adjusted_volume"]), { notation: "compact", maximumFractionDigits: 2 }));
    setText("quote-date", `As of ${dateValue(firstDefined(record, ["date", "timestamp", "eod_date"], meta.as_of))}`);
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
    return items.map((item) => Object.freeze({
      raw: item,
      date: firstDefined(item, ["date", "timestamp", "eod_date"]),
      open: firstDefined(item, ["open", "adj_open", "adjusted_open"]),
      high: firstDefined(item, ["high", "adj_high", "adjusted_high"]),
      low: firstDefined(item, ["low", "adj_low", "adjusted_low"]),
      close: firstDefined(item, ["close", "adj_close", "adjusted_close", "price"]),
      volume: firstDefined(item, ["volume", "adj_volume", "adjusted_volume"]),
    })).filter((item) => item.date && Number.isFinite(Number(item.close))).sort((a, b) => new Date(a.date) - new Date(b.date));
  }

  function renderHistoryTable(records, currency) {
    const rows = records.map((record) => {
      const row = document.createElement("tr");
      const values = [dateValue(record.date), formatMoney(record.open, currency), formatMoney(record.high, currency), formatMoney(record.low, currency), formatMoney(record.close, currency), formatNumber(record.volume, { notation: "compact", maximumFractionDigits: 2 })];
      values.forEach((value) => {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      });
      return row;
    });
    $("history-table-body").replaceChildren(...rows);
  }

  function renderChart(records, currency) {
    if (state.chart) state.chart.destroy();
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const context = $("price-chart").getContext("2d");
    if (!window.Chart) {
      $("history-table-details").open = true;
      $("price-chart").hidden = true;
      return;
    }
    $("price-chart").hidden = false;
    const chart = new window.Chart(context, {
      type: "line",
      data: {
        labels: records.map((record) => dateValue(record.date)),
        datasets: [{
          label: `${state.symbol} adjusted close`,
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
        animation: reducedMotion ? false : { duration: 220 },
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { display: true, align: "start", labels: { color: "#334155", usePointStyle: true, boxWidth: 8 } },
          tooltip: { callbacks: { label: (contextValue) => `${contextValue.dataset.label}: ${formatMoney(contextValue.parsed.y, currency)}` } },
        },
        scales: {
          x: { title: { display: true, text: "Trading date", color: "#475569" }, grid: { display: false }, ticks: { color: "#475569", maxTicksLimit: window.innerWidth < 600 ? 4 : 9, maxRotation: 0 } },
          y: { title: { display: true, text: `Price (${currency})`, color: "#475569" }, grid: { color: "#e2e8f0" }, ticks: { color: "#475569", callback: (value) => formatNumber(value, { maximumFractionDigits: 2 }) } },
        },
      },
    });
    updateState({ chart });
  }

  function renderHistory(items) {
    const records = normalizeHistory(items);
    if (!records.length) {
      setSectionState("history", "empty");
      return;
    }
    const currency = firstDefined(state.ticker, ["currency", "currency_code"], "USD");
    const first = records[0];
    const last = records[records.length - 1];
    const change = Number(last.close) - Number(first.close);
    setText("chart-summary", `${state.symbol} closed at ${formatMoney(last.close, currency)} on ${dateValue(last.date)}, ${change >= 0 ? "up" : "down"} ${formatMoney(Math.abs(change), currency)} from ${dateValue(first.date)} across ${records.length} observations.`);
    renderHistoryTable(records, currency);
    renderChart(records, currency);
    setSectionState("history", "ready");
  }

  function appendRows(targetId, records, valueFactory) {
    const rows = records.slice(0, 20).map((record) => {
      const row = document.createElement("tr");
      valueFactory(record).forEach((value) => {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      });
      return row;
    });
    $(targetId).replaceChildren(...rows);
  }

  function renderSplits(items) {
    if (!items.length) return setSectionState("splits", "empty");
    appendRows("splits-table-body", items, (item) => {
      const factor = firstDefined(item, ["split_factor", "factor", "ratio"]);
      const numerator = firstDefined(item, ["numerator", "to"]);
      const denominator = firstDefined(item, ["denominator", "from"]);
      return [dateValue(firstDefined(item, ["date", "split_date"])), factor || (numerator && denominator ? `${numerator}:${denominator}` : "—")];
    });
    setSectionState("splits", "ready");
  }

  function renderDividends(items) {
    if (!items.length) return setSectionState("dividends", "empty");
    const currency = firstDefined(state.ticker, ["currency", "currency_code"], "USD");
    appendRows("dividends-table-body", items, (item) => [dateValue(firstDefined(item, ["date", "payment_date", "ex_date"])), formatMoney(firstDefined(item, ["dividend", "amount", "value"]), firstDefined(item, ["currency"], currency))]);
    setSectionState("dividends", "ready");
  }

  function renderUsage(data) {
    const usage = nestedData(data) || {};
    const used = Number(firstDefined(usage, ["used", "requests_used", "count", "usage"], 0));
    const limit = Number(firstDefined(usage, ["limit", "requests_limit", "quota", "monthly_limit"], 0));
    const remaining = Number(firstDefined(usage, ["remaining", "requests_remaining"]));
    const resolvedUsed = Number.isFinite(used) ? used : Number.isFinite(remaining) && limit ? limit - remaining : 0;
    const percent = limit > 0 ? Math.min(100, Math.max(0, (resolvedUsed / limit) * 100)) : 0;
    setText("usage-used", formatNumber(resolvedUsed));
    setText("usage-limit", limit ? formatNumber(limit) : "unlimited");
    setText("usage-reset", firstDefined(usage, ["reset_at", "reset_date", "period_end"]) ? `Resets ${dateValue(firstDefined(usage, ["reset_at", "reset_date", "period_end"]))}` : "Reset date unavailable");
    const progress = $("usage-progress");
    progress.setAttribute("aria-valuenow", String(Math.round(percent)));
    progress.classList.toggle("is-warning", percent >= 75 && percent < 90);
    progress.classList.toggle("is-critical", percent >= 90);
    $("usage-bar").style.width = `${percent}%`;
    setSectionState("usage", "ready");
  }

  async function loadHistory(symbol) {
    setSectionState("history", "loading");
    const params = new URLSearchParams({ date_from: elements.dateFrom.value, date_to: elements.dateTo.value, limit: "1000" });
    try {
      const response = await request(`${API.history(symbol)}?${params.toString()}`);
      renderHistory(asList(response.data));
    } catch (error) {
      setSectionState("history", "error", `${error.message} Adjust the range or retry.`);
    }
  }

  async function loadSecurity(symbol) {
    updateState({ symbol });
    setText("header-symbol", symbol);
    setText("company-symbol", symbol);
    ["quote", "history", "splits", "dividends"].forEach((name) => setSectionState(name, "loading"));
    displayGlobal("");
    const historyParams = new URLSearchParams({ date_from: elements.dateFrom.value, date_to: elements.dateTo.value, limit: "1000" });
    const requests = await Promise.allSettled([
      request(API.latest(symbol)),
      request(`${API.history(symbol)}?${historyParams.toString()}`),
      request(API.splits(symbol)),
      request(API.dividends(symbol)),
    ]);
    const [latest, history, splits, dividends] = requests;
    if (latest.status === "fulfilled") {
      const record = Array.isArray(latest.value.data) ? latest.value.data[0] : latest.value.data;
      record ? renderQuote(record, latest.value.meta) : setSectionState("quote", "empty");
    } else setSectionState("quote", "error", latest.reason.message);
    if (history.status === "fulfilled") renderHistory(asList(history.value.data));
    else setSectionState("history", "error", history.reason.message);
    if (splits.status === "fulfilled") renderSplits(asList(splits.value.data));
    else setSectionState("splits", "error", splits.reason.message);
    if (dividends.status === "fulfilled") renderDividends(asList(dividends.value.data));
    else setSectionState("dividends", "error", dividends.reason.message);
  }

  async function hydrateInitialMetadata() {
    setSectionState("metadata", "loading");
    try {
      const params = new URLSearchParams({ search: state.symbol, limit: "5" });
      const response = await request(`${API.tickers}?${params.toString()}`);
      const ticker = asList(response.data).map(normalizeTicker).find((item) => item.symbol === state.symbol);
      if (ticker) {
        updateState({ ticker: ticker.raw });
        renderMetadata(ticker.raw);
      } else {
        renderMetadata({ symbol: state.symbol, name: state.symbol });
      }
    } catch (error) {
      setSectionState("metadata", "error", error.message);
    }
  }

  async function loadSupportingData() {
    setSectionState("usage", "loading");
    const [exchanges, usage] = await Promise.allSettled([request(API.exchanges), request(API.usage)]);
    if (exchanges.status === "fulfilled") {
      updateState({ exchanges: asList(exchanges.value.data) });
      if (state.ticker) renderMetadata(state.ticker);
    }
    if (usage.status === "fulfilled") renderUsage(usage.value.data);
    else setSectionState("usage", "error", usage.reason.message);
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

  function bindEvents() {
    elements.searchForm.addEventListener("submit", (event) => {
      event.preventDefault();
      if (state.searchIndex >= 0 && state.searchResults[state.searchIndex]) selectTicker(state.searchResults[state.searchIndex]);
      else runSearch(elements.searchInput.value);
    });
    elements.searchInput.addEventListener("keydown", (event) => {
      if (event.key === "ArrowDown") { event.preventDefault(); moveSearchFocus(1); }
      else if (event.key === "ArrowUp") { event.preventDefault(); moveSearchFocus(-1); }
      else if (event.key === "Escape") closeSearchResults();
    });
    document.addEventListener("click", (event) => {
      if (!elements.searchForm.contains(event.target)) closeSearchResults();
    });
    elements.historyForm.addEventListener("submit", (event) => {
      event.preventDefault();
      if (elements.dateFrom.value > elements.dateTo.value) {
        setSectionState("history", "error", "The start date must be before the end date.");
        elements.dateFrom.focus();
        return;
      }
      document.querySelectorAll(".range-button").forEach((button) => { button.classList.remove("is-active"); button.setAttribute("aria-pressed", "false"); });
      loadHistory(state.symbol);
    });
    document.querySelectorAll(".range-button").forEach((button) => {
      button.addEventListener("click", () => {
        initializeDates(Number(button.dataset.days));
        document.querySelectorAll(".range-button").forEach((candidate) => { candidate.classList.toggle("is-active", candidate === button); candidate.setAttribute("aria-pressed", candidate === button ? "true" : "false"); });
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

  async function initialize() {
    initializeDates();
    bindEvents();
    await Promise.all([hydrateInitialMetadata(), loadSupportingData(), loadSecurity(state.symbol)]);
  }

  initialize();
})();
