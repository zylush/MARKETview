"""End-to-end coverage for authentication and the primary dashboard journey."""

from __future__ import annotations

import httpx
import pytest

pytestmark = pytest.mark.e2e


def _sign_in(page, base_url: str, password: str) -> None:
    page.goto(f"{base_url}/")
    page.get_by_label("Password").fill(password)
    page.get_by_role("button", name="Sign in").click()
    page.wait_for_url(f"{base_url}/dashboard")


def test_app_key_authorizes_programmatic_api(base_url: str, app_key: str) -> None:
    response = httpx.get(
        f"{base_url}/api/v1/usage",
        headers={"X-App-Key": app_key},
        timeout=5,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["used"] == 17


def test_missing_app_key_is_rejected(base_url: str) -> None:
    response = httpx.get(f"{base_url}/api/v1/usage", timeout=5)

    assert response.status_code == 401
    payload = response.json()
    assert payload["success"] is False
    assert payload["error"]["code"] == "unauthorized"


def test_login_search_chart_metadata_and_logout(
    page,
    base_url: str,
    app_key: str,
) -> None:
    """A user can authenticate, select a security, inspect data, and sign out."""
    _sign_in(page, base_url, app_key)

    search = page.get_by_test_id("ticker-search")
    search.fill("Microsoft")
    page.get_by_test_id("search-submit").click()
    page.get_by_role("option", name="MSFT Microsoft Corporation").click()

    quote = page.get_by_test_id("quote-summary")
    quote.wait_for(state="visible")
    assert quote.get_by_text("Close", exact=True).is_visible()
    assert quote.get_by_text("$423.46", exact=True).is_visible()

    metadata = page.get_by_test_id("company-metadata")
    assert metadata.get_by_text("Microsoft Corporation", exact=True).is_visible()
    assert metadata.get_by_text("MSFT", exact=True).is_visible()
    assert metadata.get_by_text("Nasdaq Stock Market", exact=True).is_visible()

    chart = page.get_by_test_id("history-chart")
    chart.wait_for(state="visible")
    assert page.get_by_test_id("chart-summary").text_content()

    page.get_by_text("View accessible price table").click()
    history_table = page.get_by_role(
        "table", name="Historical daily open, high, low, close, and volume"
    )
    assert history_table.is_visible()
    assert history_table.locator("tbody tr").count() == 5

    assert page.get_by_test_id("splits-table").is_visible()
    assert page.get_by_test_id("dividends-table").is_visible()
    assert page.get_by_test_id("quota-usage").get_by_text("17", exact=True).is_visible()

    page.get_by_test_id("logout-button").click()
    page.wait_for_url(f"{base_url}/")
    assert page.get_by_role("heading", name="Sign in to your workspace").is_visible()


def test_dashboard_is_keyboard_operable(page, base_url: str, app_key: str) -> None:
    _sign_in(page, base_url, app_key)

    search = page.get_by_test_id("ticker-search")
    search.fill("A")
    page.get_by_test_id("search-submit").press("Enter")
    search.press("ArrowDown")
    search.press("Enter")

    page.get_by_test_id("quote-summary").wait_for(state="visible")
    assert page.locator("body").evaluate("element => element.scrollWidth <= element.clientWidth")
