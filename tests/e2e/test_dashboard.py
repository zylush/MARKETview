"""End-to-end coverage for authentication and the direct-symbol dashboard."""

from __future__ import annotations

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
    assert page.get_by_test_id("company-metadata").count() == 0
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
