"""End-to-end coverage for authentication and the symbol dashboard."""

from __future__ import annotations

import json
from urllib.parse import urlparse

import httpx
import pytest

pytestmark = pytest.mark.e2e


def _sign_in(page, base_url: str, password: str) -> None:
    page.goto(f"{base_url}/")
    page.get_by_label("Password").fill(password)
    page.get_by_role("button", name="Sign in").click()
    page.wait_for_url(f"{base_url}/dashboard")


def _api_path(url: str) -> str:
    parsed = urlparse(url)
    return parsed.path if parsed.path.startswith("/api/v1/") else ""


def test_app_key_authorizes_programmatic_api(base_url: str, app_key: str) -> None:
    response = httpx.get(
        f"{base_url}/api/v1/usage",
        headers={"X-App-Key": app_key},
        timeout=5,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["requests_used"] == 17


def test_missing_app_key_is_rejected(base_url: str) -> None:
    response = httpx.get(f"{base_url}/api/v1/usage", timeout=5)

    assert response.status_code == 401
    payload = response.json()
    assert payload["success"] is False
    assert payload["error"]["code"] == "unauthorized"


def test_inquiry_contact_is_available_on_login_and_dashboard(
    page,
    base_url: str,
    app_key: str,
) -> None:
    page.goto(f"{base_url}/")
    login_contact = page.get_by_role("link", name="paoloinigo30@gmail.com")
    assert login_contact.is_visible()
    assert login_contact.get_attribute("href") == "mailto:paoloinigo30@gmail.com"
    assert page.get_by_text(
        "Contact paoloinigo30@gmail.com for inquiries.", exact=True
    ).is_visible()

    _sign_in(page, base_url, app_key)
    dashboard_contact = page.get_by_role("link", name="paoloinigo30@gmail.com")
    assert dashboard_contact.is_visible()
    assert dashboard_contact.get_attribute("href") == "mailto:paoloinigo30@gmail.com"
    assert page.get_by_text(
        "Contact paoloinigo30@gmail.com for inquiries.", exact=True
    ).is_visible()


def test_login_landing_is_static_accessible_and_responsive(page, base_url: str) -> None:
    api_requests: list[str] = []
    page.on("request", lambda request: api_requests.append(_api_path(request.url)))
    page.set_viewport_size({"width": 375, "height": 800})

    page.goto(f"{base_url}/")

    assert page.get_by_role("heading", name="A clearer view of the market.").is_visible()
    assert page.get_by_text("MarketView", exact=True).is_visible()
    assert page.get_by_text("Private workspace", exact=True).is_visible()
    assert page.get_by_test_id("landing-preview").is_visible()
    assert page.get_by_text("Illustrative snapshot", exact=True).is_visible()
    assert page.get_by_role("heading", name="Sign in to your workspace").is_visible()
    assert page.get_by_label("Password").is_visible()
    assert page.get_by_role("button", name="Sign in").is_visible()
    assert page.locator("body").evaluate("element => element.scrollWidth <= element.clientWidth")
    assert page.get_by_label("Password").bounding_box()["height"] >= 44
    assert page.get_by_role("button", name="Sign in").bounding_box()["height"] >= 44
    assert [path for path in api_requests if path] == []


def test_login_symbol_quote_history_usage_and_logout(
    page,
    base_url: str,
    app_key: str,
) -> None:
    """A user can enter a symbol, inspect supported data, and sign out."""
    api_requests: list[str] = []
    page.on("request", lambda request: api_requests.append(_api_path(request.url)))
    _sign_in(page, base_url, app_key)

    page.get_by_test_id("quote-summary").wait_for(state="visible")
    initial_paths = [path for path in api_requests if path]
    assert initial_paths[-1] == "/api/v1/usage"
    assert sorted(initial_paths) == sorted(
        [
            "/api/v1/eod/latest/AAPL",
            "/api/v1/eod/history/AAPL",
            "/api/v1/usage",
        ]
    )

    api_requests.clear()
    search = page.get_by_test_id("ticker-search")
    search.fill("msft")
    search.press("Enter")

    quote = page.get_by_test_id("quote-summary")
    quote.wait_for(state="visible")
    assert quote.get_by_text("Unadjusted close", exact=True).is_visible()
    assert quote.get_by_text("$423.46", exact=True).is_visible()
    symbol_paths = [path for path in api_requests if path]
    assert symbol_paths[-1] == "/api/v1/usage"
    assert sorted(symbol_paths) == sorted(
        [
            "/api/v1/eod/latest/MSFT",
            "/api/v1/eod/history/MSFT",
            "/api/v1/usage",
        ]
    )

    chart = page.get_by_test_id("history-chart")
    chart.wait_for(state="visible")
    assert page.get_by_test_id("chart-summary").text_content()
    page.get_by_text("View accessible price table").click()
    history_table = page.get_by_role(
        "table", name="Historical daily open, high, low, close, and volume"
    )
    assert history_table.is_visible()
    assert history_table.locator("tbody tr").count() == 5
    assert page.get_by_test_id("quota-usage").get_by_text("17", exact=True).is_visible()
    assert page.get_by_test_id("splits-table").count() == 0
    assert page.get_by_test_id("dividends-table").count() == 0

    page.get_by_test_id("logout-button").click()
    page.wait_for_url(f"{base_url}/")
    assert page.get_by_role("heading", name="Sign in to your workspace").is_visible()


def test_invalid_symbol_is_accessible_and_makes_zero_api_requests(
    page,
    base_url: str,
    app_key: str,
) -> None:
    _sign_in(page, base_url, app_key)
    page.get_by_test_id("quote-summary").wait_for(state="visible")
    api_requests: list[str] = []
    page.on("request", lambda request: api_requests.append(_api_path(request.url)))

    search = page.get_by_test_id("ticker-search")
    search.fill("bad symbol!")
    search.press("Enter")

    error = page.get_by_role("alert").filter(has_text="Enter 1")
    assert error.is_visible()
    assert search.get_attribute("aria-invalid") == "true"
    assert [path for path in api_requests if path] == []
    assert page.locator("body").evaluate("element => element.scrollWidth <= element.clientWidth")


def test_symbol_autocomplete_is_debounced_and_keyboard_operable(
    page,
    base_url: str,
    app_key: str,
) -> None:
    api_requests: list[str] = []
    page.on("request", lambda request: api_requests.append(request.url))
    _sign_in(page, base_url, app_key)
    page.get_by_test_id("quote-summary").wait_for(state="visible")
    api_requests.clear()

    search = page.get_by_test_id("ticker-search")
    listbox = page.get_by_role("listbox", name="Ticker suggestions")
    status = page.get_by_test_id("ticker-search-status")

    assert search.get_attribute("role") == "combobox"
    assert search.get_attribute("aria-autocomplete") == "list"
    assert search.get_attribute("aria-controls") == "ticker-suggestions"
    assert page.locator("#ticker-suggestions").get_attribute("role") == "listbox"
    assert search.get_attribute("aria-expanded") == "false"
    assert status.get_attribute("role") == "status"
    assert status.get_attribute("aria-live") == "polite"

    search.fill("a")
    page.wait_for_timeout(400)
    assert not any("/api/v1/symbols/search" in url for url in api_requests)

    search.fill("ap")
    page.wait_for_timeout(100)
    assert not any("/api/v1/symbols/search" in url for url in api_requests)
    listbox.wait_for(state="visible")
    assert search.get_attribute("aria-expanded") == "true"
    assert listbox.get_by_role("option").count() <= 8
    assert listbox.get_by_role("option", name="AAPL Apple Inc. Nasdaq").is_visible()

    search.press("ArrowUp")
    active_id = search.get_attribute("aria-activedescendant")
    assert active_id
    assert page.locator(f"#{active_id}").get_by_text("APP", exact=True).is_visible()
    search.press("ArrowDown")
    active_id = search.get_attribute("aria-activedescendant")
    assert active_id
    assert page.locator(f"#{active_id}").get_attribute("aria-selected") == "true"
    search.press("Escape")
    assert listbox.is_hidden()
    assert search.get_attribute("aria-expanded") == "false"
    assert search.get_attribute("aria-activedescendant") is None

    search.fill("ap")
    listbox.wait_for(state="visible")
    search.press("ArrowDown")
    search.press("ArrowDown")
    search.press("ArrowUp")
    search.press("Enter")
    page.get_by_text("AAPL", exact=True).first.wait_for(state="visible")
    assert search.input_value() == "AAPL"
    assert listbox.is_hidden()
    assert any("/api/v1/eod/latest/AAPL" in url for url in api_requests)
    autocomplete_urls = [url for url in api_requests if "/api/v1/symbols/search" in url]
    assert autocomplete_urls
    assert all("q=AP" in url for url in autocomplete_urls)
    assert all("limit=8" in url for url in autocomplete_urls)


def test_symbol_autocomplete_aborts_stale_requests(page, base_url: str, app_key: str) -> None:
    page.add_init_script(
        """
        const NativeAbortController = window.AbortController;
        window.autocompleteAbortCount = 0;
        window.AbortController = class TrackedAbortController extends NativeAbortController {
          constructor() {
            super();
            this.signal.addEventListener('abort', () => { window.autocompleteAbortCount += 1; });
          }
        };
        const originalFetch = window.fetch.bind(window);
        window.fetch = async (...args) => {
          if (String(args[0]).includes('/api/v1/symbols/search?q=AP')) {
            await new Promise((resolve) => setTimeout(resolve, 500));
          }
          return originalFetch(...args);
        };
        """
    )
    _sign_in(page, base_url, app_key)
    page.get_by_test_id("quote-summary").wait_for(state="visible")
    search = page.get_by_test_id("ticker-search")
    status = page.get_by_test_id("ticker-search-status")

    search.fill("ap")
    status.get_by_text("Loading ticker suggestions", exact=False).wait_for(state="visible")
    search.fill("ms")
    option = page.get_by_role("option", name="MSFT Microsoft Corporation Nasdaq")
    option.wait_for(state="visible")
    page.wait_for_timeout(600)

    assert page.evaluate("window.autocompleteAbortCount") >= 1
    assert option.is_visible()
    assert page.get_by_role("option", name="AAPL Apple Inc. Nasdaq").count() == 0


def test_symbol_autocomplete_supports_mouse_and_announces_request_states(
    page,
    base_url: str,
    app_key: str,
) -> None:
    page.add_init_script(
        """
        const originalFetch = window.fetch.bind(window);
        window.fetch = async (...args) => {
          if (String(args[0]).includes('/api/v1/symbols/search')) {
            await new Promise((resolve) => setTimeout(resolve, 400));
          }
          return originalFetch(...args);
        };
        """
    )
    _sign_in(page, base_url, app_key)
    page.get_by_test_id("quote-summary").wait_for(state="visible")

    search = page.get_by_test_id("ticker-search")
    status = page.get_by_test_id("ticker-search-status")
    listbox = page.get_by_role("listbox", name="Ticker suggestions")
    search.fill("app")
    status.get_by_text("Loading ticker suggestions", exact=False).wait_for(state="visible")
    listbox.wait_for(state="visible")

    listbox.get_by_role("option", name="APP Applovin Corporation Nasdaq").click()
    assert search.input_value() == "APP"
    page.get_by_text("APP", exact=True).first.wait_for(state="visible")
    assert listbox.is_hidden()


def test_symbol_autocomplete_announces_empty_and_error_states(
    page,
    base_url: str,
    app_key: str,
) -> None:
    def symbol_response(route) -> None:
        query = urlparse(route.request.url).query
        if "q=ER" in query:
            route.fulfill(
                status=503,
                content_type="application/json",
                body=json.dumps(
                    {
                        "success": False,
                        "data": None,
                        "error": {"message": "Directory temporarily unavailable."},
                    }
                ),
            )
            return
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "success": True,
                    "data": {"items": [], "limit": 8, "total": 0},
                    "error": None,
                }
            ),
        )

    page.route("**/api/v1/symbols/search**", symbol_response)
    _sign_in(page, base_url, app_key)
    page.get_by_test_id("quote-summary").wait_for(state="visible")
    search = page.get_by_test_id("ticker-search")
    status = page.get_by_test_id("ticker-search-status")

    search.fill("zz")
    status.get_by_text("No ticker suggestions found", exact=False).wait_for(state="visible")
    assert search.get_attribute("aria-expanded") == "false"

    search.fill("er")
    status.get_by_text("Could not load ticker suggestions", exact=False).wait_for(state="visible")
    assert search.get_attribute("aria-expanded") == "false"


def test_symbol_autocomplete_renders_external_values_as_text_without_overflow(
    page,
    base_url: str,
    app_key: str,
) -> None:
    malicious_name = '<img src=x onerror="window.autocompleteXss=true">'

    def symbol_response(route) -> None:
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "success": True,
                    "data": {
                        "items": [
                            {
                                "symbol": "XSS",
                                "name": malicious_name,
                                "exchange": "NYSE",
                            }
                        ],
                        "limit": 8,
                        "total": 1,
                    },
                    "error": None,
                }
            ),
        )

    page.route("**/api/v1/symbols/search**", symbol_response)
    page.set_viewport_size({"width": 375, "height": 667})
    _sign_in(page, base_url, app_key)
    page.get_by_test_id("quote-summary").wait_for(state="visible")

    search = page.get_by_test_id("ticker-search")
    search.fill("xs")
    option = page.get_by_role("option", name=f"XSS {malicious_name} NYSE")
    option.wait_for(state="visible")
    assert option.locator("img").count() == 0
    assert option.get_by_text(malicious_name, exact=True).is_visible()
    assert option.bounding_box()["height"] >= 44
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    assert page.evaluate("window.autocompleteXss !== true")


def test_marketview_panel_is_bounded_accessible_posts_with_csrf_and_renders_metadata(
    page,
    base_url: str,
    app_key: str,
) -> None:
    research_requests: list[dict[str, object]] = []

    def record_research(request) -> None:
        if "/api/v1/research/query" in request.url:
            research_requests.append(
                {
                    "method": request.method,
                    "headers": request.headers,
                    "body": request.post_data_json,
                }
            )

    def marketview_response(route) -> None:
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "success": True,
                    "data": {
                        "symbol": "AAPL",
                        "status": "answered",
                        "answer": "AAPL closed higher over the selected period.",
                        "provider": "marketdata.app",
                        "as_of": "2026-08-08T00:00:00Z",
                        "period_start": "2026-07-09",
                        "period_end": "2026-08-07",
                        "evidence_count": 22,
                        "disclaimer": "Informational only, not investment advice.",
                    },
                    "meta": {"request_id": "browser-fixture"},
                    "error": None,
                }
            ),
        )

    page.route("**/api/v1/research/query", marketview_response)
    page.on("request", record_research)
    _sign_in(page, base_url, app_key)
    page.get_by_test_id("quote-summary").wait_for(state="visible")

    panel = page.get_by_test_id("research-panel")
    question = page.get_by_label("Question about market data")
    submit = page.get_by_test_id("research-submit")
    counter = page.get_by_test_id("research-character-count")

    assert panel.get_by_role("heading", name="MarketView AI").is_visible()
    assert panel.get_by_text("Informational only, not investment advice.", exact=True).is_visible()
    assert question.get_attribute("maxlength") == "500"
    assert question.get_attribute("aria-describedby") == "research-help research-character-count"
    assert counter.get_attribute("aria-live") == "polite"
    assert counter.text_content() == "0 / 500"
    assert submit.is_disabled()

    prompt = "How has the closing price changed recently?"
    question.fill(prompt)
    assert counter.text_content() == f"{len(prompt)} / 500"
    assert submit.is_enabled()
    submit.click()

    panel.get_by_text("AAPL closed higher over the selected period.", exact=True).wait_for(
        state="visible"
    )
    assert panel.get_by_text("Answered", exact=True).is_visible()
    assert panel.get_by_text("marketdata.app", exact=True).is_visible()
    assert panel.get_by_text("Jul 9, 2026", exact=False).is_visible()
    assert panel.get_by_text("Aug 7, 2026", exact=False).is_visible()
    assert panel.get_by_text("22", exact=True).is_visible()
    assert page.evaluate("document.activeElement.id") == "research-result"
    assert research_requests == [
        {
            "method": "POST",
            "headers": research_requests[0]["headers"],
            "body": {"symbol": "AAPL", "question": prompt},
        }
    ]
    headers = research_requests[0]["headers"]
    assert headers["content-type"] == "application/json"
    assert headers["x-csrf-token"]
    assert headers["x-csrf-token"] == page.locator("meta[name='csrf-token']").get_attribute(
        "content"
    )


def test_marketview_renders_answered_insufficient_refused_timeout_and_unavailable_states(
    page,
    base_url: str,
    app_key: str,
) -> None:
    def marketview_response(route) -> None:
        question_text = route.request.post_data_json["question"].lower()
        if "timeout" in question_text or "unavailable" in question_text:
            timeout = "timeout" in question_text
            route.fulfill(
                status=504 if timeout else 503,
                content_type="application/json",
                body=json.dumps(
                    {
                        "success": False,
                        "data": None,
                        "meta": {"request_id": "browser-fixture"},
                        "error": {
                            "code": "RESEARCH_TIMEOUT" if timeout else "RESEARCH_UNAVAILABLE",
                            "message": "market analysis could not be completed",
                        },
                    }
                ),
            )
            return
        status = "answered"
        answer = "AAPL gained 3.2% over the selected period."
        if "missing" in question_text:
            status = "insufficient_evidence"
            answer = ""
        elif "buy" in question_text:
            status = "refused"
            answer = ""
        elif "unknown" in question_text:
            status = "future_status"
            answer = "THIS ANSWER MUST NOT RENDER"
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "success": True,
                    "data": {
                        "symbol": "AAPL",
                        "status": status,
                        "answer": answer,
                        "provider": "marketdata.app",
                        "as_of": "2026-08-08T00:00:00Z",
                        "period_start": "2026-07-09",
                        "period_end": "2026-08-07",
                        "evidence_count": 22,
                        "disclaimer": "Informational only, not investment advice.",
                    },
                    "error": None,
                }
            ),
        )

    page.route("**/api/v1/research/query", marketview_response)
    _sign_in(page, base_url, app_key)
    page.get_by_test_id("quote-summary").wait_for(state="visible")
    panel = page.get_by_test_id("research-panel")
    question = page.get_by_label("Question about market data")
    submit = page.get_by_test_id("research-submit")

    question.fill("How has the closing price changed recently?")
    submit.click()
    panel.get_by_text("AAPL gained 3.2% over the selected period.", exact=True).wait_for(
        state="visible"
    )

    question.fill("What evidence is missing?")
    submit.click()
    panel.get_by_text("Insufficient evidence", exact=False).wait_for(state="visible")

    question.fill("Should I buy this stock?")
    submit.click()
    panel.get_by_text(
        "cannot provide personalized buy or sell recommendations", exact=False
    ).wait_for(state="visible")

    question.fill("Return an unknown status")
    submit.click()
    panel.get_by_text("Market analysis is temporarily unavailable.", exact=True).wait_for(
        state="visible"
    )
    assert panel.get_by_text("THIS ANSWER MUST NOT RENDER", exact=True).count() == 0

    question.fill("Please timeout")
    submit.click()
    panel.get_by_text("Market analysis timed out", exact=False).wait_for(state="visible")
    question.fill("Provider unavailable")
    submit.click()
    panel.get_by_text("Market analysis is temporarily unavailable", exact=False).wait_for(
        state="visible"
    )


def test_marketview_cancel_and_symbol_change_abort_and_ignore_stale_completion(
    page,
    base_url: str,
    app_key: str,
) -> None:
    def delayed_marketview_response(route) -> None:
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "success": True,
                    "data": {
                        "symbol": "AAPL",
                        "status": "answered",
                        "answer": "UNIQUE STALE AAPL ANALYSIS",
                        "provider": "stale-provider",
                        "as_of": "2026-08-08T00:00:00Z",
                        "period_start": "2026-07-09",
                        "period_end": "2026-08-07",
                        "evidence_count": 22,
                        "disclaimer": "Informational only, not investment advice.",
                    },
                    "error": None,
                }
            ),
        )

    page.add_init_script(
        """
        const originalFetch = window.fetch.bind(window);
        window.researchAbortCount = 0;
        window.fetch = (url, options = {}) => {
          if (!String(url).includes('/api/v1/research/query')) return originalFetch(url, options);
          return new Promise((resolve, reject) => {
            const timer = setTimeout(() => resolve(originalFetch(url, options)), 700);
            options.signal?.addEventListener('abort', () => {
              clearTimeout(timer);
              window.researchAbortCount += 1;
              reject(new DOMException('Aborted', 'AbortError'));
            }, { once: true });
          });
        };
        """
    )
    page.route("**/api/v1/research/query", delayed_marketview_response)
    _sign_in(page, base_url, app_key)
    page.get_by_test_id("quote-summary").wait_for(state="visible")
    panel = page.get_by_test_id("research-panel")
    question = page.get_by_label("Question about market data")
    submit = page.get_by_test_id("research-submit")
    cancel = page.get_by_test_id("research-cancel")

    question.fill("How has the closing price changed recently?")
    submit.click()
    panel.get_by_text("Analyzing market data for AAPL", exact=False).wait_for(state="visible")
    assert submit.is_disabled()
    assert cancel.is_visible()
    cancel.click()
    panel.get_by_text("Analysis request cancelled", exact=False).wait_for(state="visible")
    assert page.evaluate("window.researchAbortCount") == 1

    submit.click()
    panel.get_by_text("Analyzing market data for AAPL", exact=False).wait_for(state="visible")
    question.press("Escape")
    panel.get_by_text("Analysis request cancelled", exact=False).wait_for(state="visible")
    assert page.evaluate("window.researchAbortCount") == 2

    submit.click()
    panel.get_by_text("Analyzing market data for AAPL", exact=False).wait_for(state="visible")
    search = page.get_by_test_id("ticker-search")
    search.fill("msft")
    search.press("Enter")
    page.locator("#company-symbol").get_by_text("MSFT", exact=True).wait_for(state="visible")
    panel.get_by_text("Ask MarketView about MSFT", exact=False).wait_for(state="visible")
    page.wait_for_timeout(850)
    assert page.evaluate("window.researchAbortCount") >= 3
    assert panel.get_by_text("UNIQUE STALE AAPL ANALYSIS", exact=True).count() == 0
    assert panel.get_by_text("stale-provider", exact=True).count() == 0
    assert question.input_value() == ""


def test_marketview_renders_partial_market_analysis_metadata(
    page,
    base_url: str,
    app_key: str,
) -> None:
    def combined_response(route) -> None:
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "success": True,
                    "data": {
                        "symbol": "AAPL",
                        "status": "partial",
                        "answer": "AAPL's available records show a closing price of 204.50.",
                        "provider": "marketdata.app",
                        "as_of": "2026-08-08T00:00:00Z",
                        "period_start": "2026-08-07",
                        "period_end": "2026-08-07",
                        "evidence_count": 1,
                        "disclaimer": "Informational only, not investment advice.",
                    },
                    "meta": {"request_id": "browser-fixture"},
                    "error": None,
                }
            ),
        )

    page.route("**/api/v1/research/query", combined_response)
    _sign_in(page, base_url, app_key)
    page.get_by_test_id("quote-summary").wait_for(state="visible")
    question = page.get_by_label("Question about market data")
    question.fill("What is the latest closing price?")
    page.get_by_test_id("research-submit").click()

    panel = page.get_by_test_id("research-panel")
    panel.get_by_text("AAPL's available records show a closing price of 204.50.").wait_for(
        state="visible"
    )
    assert panel.get_by_text("Partial", exact=True).is_visible()
    assert panel.get_by_text("marketdata.app", exact=True).is_visible()
    assert panel.get_by_text("1", exact=True).is_visible()


def test_marketview_escapes_all_response_fields_and_fits_375px(
    page,
    base_url: str,
    app_key: str,
) -> None:
    malicious = '<img src=x onerror="window.researchXss=true">'

    def research_response(route) -> None:
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "success": True,
                    "data": {
                        "symbol": "AAPL",
                        "status": "answered",
                        "answer": malicious,
                        "provider": malicious,
                        "as_of": "2026-08-08T00:00:00Z",
                        "period_start": "2026-07-09",
                        "period_end": "2026-08-07",
                        "evidence_count": 22,
                        "disclaimer": malicious,
                    },
                    "meta": {"request_id": "browser-fixture"},
                    "error": None,
                }
            ),
        )

    page.route("**/api/v1/research/query", research_response)
    page.set_viewport_size({"width": 375, "height": 667})
    _sign_in(page, base_url, app_key)
    page.get_by_test_id("quote-summary").wait_for(state="visible")
    panel = page.get_by_test_id("research-panel")
    question = page.get_by_label("Question about market data")
    question.fill("How has the closing price changed recently?")
    page.get_by_test_id("research-submit").click()

    panel.get_by_text(malicious, exact=True).first.wait_for(state="visible")
    assert panel.locator("img").count() == 0
    assert page.evaluate("window.researchXss !== true")
    assert panel.get_by_role("link").count() == 0
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    assert question.bounding_box()["height"] >= 44
    assert page.get_by_test_id("research-submit").bounding_box()["height"] >= 44
