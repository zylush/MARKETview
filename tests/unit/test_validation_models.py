from datetime import date

import pytest
from pydantic import ValidationError

from app.errors import InputValidationError
from app.models import EODBar, Page
from app.validation import validate_cursor, validate_date_range, validate_limit, validate_symbol


def test_validation_accepts_and_normalizes_valid_inputs() -> None:
    assert validate_symbol(" msft ") == "MSFT"
    assert validate_limit(100) == 100
    assert validate_cursor("120") == "120"
    assert validate_date_range(date(2025, 1, 1), date(2025, 12, 31)) == (
        date(2025, 1, 1),
        date(2025, 12, 31),
    )


@pytest.mark.parametrize("symbol", ["", "../secret", "A B", "A" * 33])
def test_symbol_validation_rejects_unsafe_values(symbol: str) -> None:
    with pytest.raises(InputValidationError, match="symbol"):
        validate_symbol(symbol)


def test_query_validation_enforces_bounds() -> None:
    with pytest.raises(ValueError, match="one year"):
        validate_date_range(date(2024, 1, 1), date(2025, 1, 2))
    with pytest.raises(InputValidationError, match="after"):
        validate_date_range(date(2025, 2, 1), date(2025, 1, 1))
    with pytest.raises(InputValidationError, match="limit"):
        validate_limit(0)
    with pytest.raises(InputValidationError, match="cursor"):
        validate_cursor("not-an-offset")


def test_domain_models_are_immutable_and_provider_neutral() -> None:
    bar = EODBar(
        symbol="MSFT",
        date="2025-01-02",
        open=100,
        high=110,
        low=99,
        close=108,
        volume=1000,
    )
    page = Page[EODBar](items=(bar,), next_cursor="1")

    with pytest.raises(ValidationError):
        bar.close = 1
    assert page.items == (bar,)
