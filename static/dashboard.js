(() => {
  "use strict";

  const API = Object.freeze({
    latest: (symbol) => `/api/v1/eod/latest/${encodeURIComponent(symbol)}`,
    history: (symbol) => `/api/v1/eod/history/${encodeURIComponent(symbol)}`,
    symbols: (query) => {
      const params = new URLSearchParams({ q: query, limit: "8" });
      return `/api/v1/symbols/search?${params.toString()}`;
    },
    research: "/api/v1/research/query",
    usage: "/api/v1/usage",
  });
  const SYMBOL_PATTERN = /^[A-Za-z0-9][A-Za-z0-9.-]{0,31}$/;
  const SYMBOL_QUERY_PATTERN = /^[A-Za-z0-9][A-Za-z0-9 .-]{1,31}$/;
  const AUTOCOMPLETE_DELAY_MS = 250;
  const $ = (id) => document.getElementById(id);
  const elements = Object.freeze({
    searchForm: $("ticker-search-form"),
    searchInput: $("ticker-search"),
    searchButton: document.querySelector("[data-testid='search-submit']"),
    searchError: $("search-error"),
    searchStatus: $("ticker-search-status"),
    suggestionList: $("ticker-suggestions"),
    historyForm: $("history-range-form"),
    dateFrom: $("date-from"),
    dateTo: $("date-to"),
    logoutForm: $("logout-form"),
    researchForm: $("research-form"),
    researchQuestion: $("research-question"),
    researchSubmit: document.querySelector("[data-testid='research-submit']"),
    researchCancel: $("research-cancel"),
  });
  let state = Object.freeze({
    symbol: String(document.body.dataset.initialSymbol || "AAPL").toUpperCase(),
    chart: null,
    autocomplete: Object.freeze({
      items: Object.freeze([]),
      activeIndex: -1,
      debounceTimer: null,
      controller: null,
      requestVersion: 0,
    }),
    research: Object.freeze({
      controller: null,
      requestVersion: 0,
      loading: false,
    }),
  });

  function updateState(changes) {
    state = Object.freeze({ ...state, ...changes });
  }

  function updateAutocomplete(changes) {
    updateState({
      autocomplete: Object.freeze({ ...state.autocomplete, ...changes }),
    });
  }

  function updateResearch(changes) {
    updateState({
      research: Object.freeze({ ...state.research, ...changes }),
    });
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

  async function request(url, options = {}) {
    const response = await fetch(url, {
      method: "GET",
      credentials: "same-origin",
      headers: { Accept: "application/json" },
      signal: options.signal,
    });
    let payload = null;
    try {
      payload = await response.json();
    } catch (_error) {
      throw new Error("The server returned an unreadable response.");
    }
    if (!response.ok || payload.success === false) {
      const error = new Error(payload?.error?.message || `Request failed (${response.status}).`);
      error.status = response.status;
      error.code = payload?.error?.code || "";
      throw error;
    }
    return payload;
  }

  async function postJson(url, body, signal) {
    const csrf = document.querySelector("meta[name='csrf-token']")?.content || "";
    const response = await fetch(url, {
      method: "POST",
      credentials: "same-origin",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
        "X-CSRF-Token": csrf,
      },
      body: JSON.stringify(body),
      signal,
    });
    let payload = null;
    try {
      payload = await response.json();
    } catch (_error) {
      const error = new Error("The server returned an unreadable response.");
      error.status = response.status;
      error.code = "UNREADABLE_RESPONSE";
      throw error;
    }
    if (!response.ok || payload?.success === false) {
      const error = new Error(payload?.error?.message || `Request failed (${response.status}).`);
      error.status = response.status;
      error.code = payload?.error?.code || "";
      throw error;
    }
    return payload;
  }

  function normalizeSymbol(value) {
    const normalized = String(value || "").trim().toUpperCase();
    return SYMBOL_PATTERN.test(normalized) ? normalized : null;
  }

  function normalizeSymbolQuery(value) {
    const normalized = String(value || "").trim().toUpperCase();
    return SYMBOL_QUERY_PATTERN.test(normalized) ? normalized : null;
  }

  function setSuggestionStatus(message, isError = false) {
    elements.searchStatus.textContent = message;
    elements.searchStatus.classList.toggle("is-error", isError);
  }

  function closeSuggestions({ clearItems = true, clearStatus = false } = {}) {
    elements.suggestionList.hidden = true;
    elements.searchInput.setAttribute("aria-expanded", "false");
    elements.searchInput.removeAttribute("aria-activedescendant");
    if (clearItems) elements.suggestionList.replaceChildren();
    if (clearStatus) setSuggestionStatus("");
    updateAutocomplete({
      items: clearItems ? Object.freeze([]) : state.autocomplete.items,
      activeIndex: -1,
    });
  }

  function cancelAutocomplete({ clearStatus = false } = {}) {
    const { debounceTimer, controller, requestVersion } = state.autocomplete;
    if (debounceTimer !== null) window.clearTimeout(debounceTimer);
    if (controller) controller.abort();
    updateAutocomplete({
      debounceTimer: null,
      controller: null,
      requestVersion: requestVersion + 1,
    });
    closeSuggestions({ clearStatus });
  }

  function normalizeSuggestion(item) {
    const symbol = normalizeSymbol(item?.symbol);
    const name = typeof item?.name === "string" ? item.name.trim() : "";
    const exchange = typeof item?.exchange === "string" ? item.exchange.trim() : "";
    if (!symbol || !name || !exchange) return null;
    return Object.freeze({ symbol, name, exchange });
  }

  function suggestionOption(item, index) {
    const option = document.createElement("div");
    const symbol = document.createElement("span");
    const name = document.createElement("span");
    const exchange = document.createElement("span");
    option.id = `ticker-suggestion-${index}`;
    option.className = "ticker-suggestion";
    option.setAttribute("role", "option");
    option.setAttribute("aria-selected", "false");
    option.dataset.index = String(index);
    symbol.className = "ticker-suggestion-symbol";
    name.className = "ticker-suggestion-name";
    exchange.className = "ticker-suggestion-exchange";
    symbol.textContent = item.symbol;
    name.textContent = item.name;
    exchange.textContent = item.exchange;
    option.replaceChildren(symbol, name, exchange);
    return option;
  }

  function renderSuggestions(rawItems) {
    const items = Object.freeze(
      rawItems.slice(0, 8).map(normalizeSuggestion).filter(Boolean),
    );
    if (!items.length) {
      closeSuggestions();
      setSuggestionStatus("No ticker suggestions found. Type a symbol and press Enter to load it directly.");
      return;
    }
    const options = items.map(suggestionOption);
    elements.suggestionList.replaceChildren(...options);
    elements.suggestionList.hidden = false;
    elements.searchInput.setAttribute("aria-expanded", "true");
    updateAutocomplete({ items, activeIndex: -1 });
    const noun = items.length === 1 ? "suggestion" : "suggestions";
    setSuggestionStatus(`${items.length} ticker ${noun} available. Use the arrow keys to review them.`);
  }

  function setActiveSuggestion(index) {
    const { items } = state.autocomplete;
    if (!items.length) return;
    const nextIndex = ((index % items.length) + items.length) % items.length;
    Array.from(elements.suggestionList.children).forEach((option, optionIndex) => {
      option.setAttribute("aria-selected", optionIndex === nextIndex ? "true" : "false");
    });
    const active = elements.suggestionList.children[nextIndex];
    elements.searchInput.setAttribute("aria-activedescendant", active.id);
    active.scrollIntoView({ block: "nearest" });
    updateAutocomplete({ activeIndex: nextIndex });
  }

  async function loadSuggestions(query) {
    const controller = new AbortController();
    const requestVersion = state.autocomplete.requestVersion + 1;
    updateAutocomplete({ controller, requestVersion, debounceTimer: null });
    setSuggestionStatus("Loading ticker suggestions…");
    try {
      const response = await request(API.symbols(query), { signal: controller.signal });
      if (controller.signal.aborted || requestVersion !== state.autocomplete.requestVersion) return;
      renderSuggestions(asList(response.data));
    } catch (error) {
      if (error?.name === "AbortError" || requestVersion !== state.autocomplete.requestVersion) return;
      closeSuggestions();
      setSuggestionStatus("Could not load ticker suggestions. Type a symbol and press Enter instead.", true);
    } finally {
      if (requestVersion === state.autocomplete.requestVersion) {
        updateAutocomplete({ controller: null });
      }
    }
  }

  function queueSuggestions() {
    clearSymbolError();
    cancelAutocomplete({ clearStatus: true });
    const query = normalizeSymbolQuery(elements.searchInput.value);
    if (!query) return;
    const debounceTimer = window.setTimeout(() => loadSuggestions(query), AUTOCOMPLETE_DELAY_MS);
    updateAutocomplete({ debounceTimer });
  }

  function selectSuggestion(index) {
    const item = state.autocomplete.items[index];
    if (!item) return;
    cancelAutocomplete({ clearStatus: true });
    elements.searchInput.value = item.symbol;
    loadDashboard(item.symbol);
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

  function updateResearchControls() {
    if (!elements.researchForm) return;
    const maximum = Number(elements.researchQuestion.maxLength) || 500;
    const length = elements.researchQuestion.value.length;
    setText("research-character-count", `${length} / ${maximum}`);
    elements.researchSubmit.disabled = state.research.loading
      || !elements.researchQuestion.value.trim()
      || length > maximum;
    setVisible(elements.researchCancel, state.research.loading);
  }

  function clearResearchOutput() {
    if (!elements.researchForm) return;
    setVisible($("research-error"), false);
    setVisible($("research-result"), false);
    setText("research-answer", "");
    setText("research-result-status", "—");
    setText("research-provider", "—");
    setText("research-as-of", "—");
    setText("research-period", "—");
    setText("research-evidence-count", "—");
    setText("research-result-disclaimer", "");
  }

  function setResearchStatus(message, { loading = false, focus = false } = {}) {
    if (!elements.researchForm) return;
    const status = $("research-status");
    status.textContent = message;
    status.classList.toggle("is-loading", loading);
    setVisible(status, true);
    if (focus) status.focus({ preventScroll: true });
  }

  function setResearchError(message) {
    clearResearchOutput();
    setVisible($("research-status"), false);
    const error = $("research-error");
    error.textContent = message;
    setVisible(error, true);
    error.focus({ preventScroll: true });
  }

  function displayStatus(value) {
    const status = typeof value === "string" ? value.trim().toLowerCase() : "";
    const labels = Object.freeze({
      answered: "Answered",
      partial: "Partial",
      insufficient_evidence: "Insufficient evidence",
      unavailable: "Unavailable",
      refused: "Refused",
    });
    return labels[status] || "Unavailable";
  }

  function safeResponseText(value, fallback = "—") {
    return typeof value === "string" && value.trim() ? value.trim() : fallback;
  }

  function renderResearchAnswer(data) {
    const result = data && typeof data === "object" ? data : {};
    if (normalizeSymbol(result.symbol) !== state.symbol) {
      setResearchError("Market analysis is temporarily unavailable. Try again later.");
      return;
    }
    const status = typeof result.status === "string"
      ? result.status.trim().toLowerCase()
      : (typeof result.outcome === "string" ? result.outcome.trim().toLowerCase() : "unavailable");
    clearResearchOutput();
    setVisible($("research-status"), false);
    const answer = $("research-answer");
    if (status === "refused" || result.refused === true) {
      answer.textContent = "I cannot provide personalized buy or sell recommendations.";
    } else if (status === "unavailable") {
      answer.textContent = "Market analysis is temporarily unavailable.";
    } else if (status === "insufficient_evidence" || result.insufficient_evidence === true) {
      answer.textContent = "There is not enough collected market data to answer this question.";
    } else if (status === "answered" || status === "partial") {
      answer.textContent = safeResponseText(
        result.answer,
        "There is not enough collected market data to answer this question.",
      );
    } else {
      answer.textContent = "Market analysis is temporarily unavailable.";
    }
    const start = typeof result.period_start === "string" ? dateValue(result.period_start) : "";
    const end = typeof result.period_end === "string" ? dateValue(result.period_end) : "";
    const period = start && end ? `${start} – ${end}` : (start || end || "—");
    const evidenceCount = result.evidence_count == null ? Number.NaN : Number(result.evidence_count);
    setText("research-result-status", displayStatus(status));
    setText("research-provider", safeResponseText(result.provider));
    setText("research-as-of", typeof result.as_of === "string" ? dateValue(result.as_of) : "—");
    setText("research-period", period);
    setText(
      "research-evidence-count",
      Number.isInteger(evidenceCount) && evidenceCount >= 0 ? formatNumber(evidenceCount) : "—",
    );
    setText(
      "research-result-disclaimer",
      safeResponseText(result.disclaimer, "AI-assisted analysis is informational only, not investment advice."),
    );
    const response = $("research-result");
    setVisible(response, true);
    response.focus({ preventScroll: true });
  }

  function cancelResearch(message = "Analysis request cancelled.") {
    if (!elements.researchForm) return;
    const { controller, requestVersion } = state.research;
    if (controller) controller.abort();
    updateResearch({
      controller: null,
      requestVersion: requestVersion + 1,
      loading: false,
    });
    clearResearchOutput();
    setResearchStatus(message, { focus: true });
    updateResearchControls();
  }

  function resetResearch(symbol) {
    if (!elements.researchForm) return;
    const { controller, requestVersion } = state.research;
    if (controller) controller.abort();
    updateResearch({
      controller: null,
      requestVersion: requestVersion + 1,
      loading: false,
    });
    elements.researchQuestion.value = "";
    setText("research-symbol", symbol);
    clearResearchOutput();
    setResearchStatus(`Ask MarketView about ${symbol} market data.`);
    updateResearchControls();
  }

  function researchErrorMessage(error) {
    if (error?.status === 504 || error?.code === "RESEARCH_TIMEOUT") {
      return "Market analysis timed out. Try again.";
    }
    if (error?.status === 503 || error?.code === "RESEARCH_UNAVAILABLE") {
      return "Market analysis is temporarily unavailable. Try again later.";
    }
    if (error?.status === 429) {
      return "Market analysis request limit reached. Try again later.";
    }
    if (error?.status === 422 || error?.status === 413) {
      return "Check the question length and try again.";
    }
    return "Could not complete the market analysis. Check your connection and try again.";
  }

  async function submitResearch() {
    if (!elements.researchForm || state.research.loading) return;
    const question = elements.researchQuestion.value.trim();
    if (!question || question.length > elements.researchQuestion.maxLength) {
      setResearchError("Enter a question within the character limit.");
      elements.researchQuestion.focus();
      return;
    }
    const controller = new AbortController();
    const requestVersion = state.research.requestVersion + 1;
    const requestedSymbol = state.symbol;
    updateResearch({ controller, requestVersion, loading: true });
    clearResearchOutput();
    setResearchStatus(`Analyzing market data for ${requestedSymbol}…`, { loading: true });
    updateResearchControls();
    try {
      const response = await postJson(
        API.research,
        Object.freeze({ symbol: requestedSymbol, question }),
        controller.signal,
      );
      if (
        controller.signal.aborted
        || requestVersion !== state.research.requestVersion
        || requestedSymbol !== state.symbol
      ) return;
      renderResearchAnswer(response.data);
    } catch (error) {
      if (error?.name === "AbortError" || requestVersion !== state.research.requestVersion) return;
      setResearchError(researchErrorMessage(error));
    } finally {
      if (requestVersion === state.research.requestVersion) {
        updateResearch({ controller: null, loading: false });
        updateResearchControls();
      }
    }
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
    const symbolChanged = state.symbol !== symbol;
    updateState({ symbol });
    if (symbolChanged) resetResearch(symbol);
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
    cancelAutocomplete({ clearStatus: true });
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
    elements.searchInput.addEventListener("input", queueSuggestions);
    elements.searchInput.addEventListener("keydown", (event) => {
      const { activeIndex, items } = state.autocomplete;
      if (event.key === "Escape" && (
        !elements.suggestionList.hidden
        || state.autocomplete.debounceTimer !== null
        || state.autocomplete.controller
      )) {
        event.preventDefault();
        cancelAutocomplete({ clearStatus: true });
        return;
      }
      if (elements.suggestionList.hidden || !items.length) return;
      if (event.key === "ArrowDown") {
        event.preventDefault();
        setActiveSuggestion(activeIndex + 1);
      } else if (event.key === "ArrowUp") {
        event.preventDefault();
        setActiveSuggestion(activeIndex < 0 ? items.length - 1 : activeIndex - 1);
      } else if (event.key === "Enter" && activeIndex >= 0) {
        event.preventDefault();
        selectSuggestion(activeIndex);
      }
    });
    elements.suggestionList.addEventListener("pointerdown", (event) => {
      if (event.target.closest("[role='option']")) event.preventDefault();
    });
    elements.suggestionList.addEventListener("click", (event) => {
      const option = event.target.closest("[role='option']");
      if (option) selectSuggestion(Number(option.dataset.index));
    });
    document.addEventListener("pointerdown", (event) => {
      if (!elements.searchForm.contains(event.target)) cancelAutocomplete({ clearStatus: true });
    });
    if (elements.researchForm) {
      elements.researchQuestion.addEventListener("input", updateResearchControls);
      elements.researchQuestion.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && state.research.loading) {
          event.preventDefault();
          cancelResearch();
        }
      });
      elements.researchForm.addEventListener("submit", (event) => {
        event.preventDefault();
        submitResearch();
      });
      elements.researchCancel.addEventListener("click", () => cancelResearch());
    }
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
    updateResearchControls();
    const initial = normalizeSymbol(state.symbol) || "AAPL";
    loadDashboard(initial);
  }

  initialize();
})();
