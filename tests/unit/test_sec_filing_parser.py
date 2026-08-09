from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from app.providers.sec_filing_parser import SecFilingParser
from app.research.domain import FilingReference, RawFiling

FIXTURES = Path(__file__).parents[1] / "fixtures" / "sec_filings"


def _parser_traceback_values(error: BaseException) -> tuple[object, ...]:
    retained: list[object] = []
    traceback = error.__traceback__
    while traceback is not None:
        if Path(traceback.tb_frame.f_code.co_filename).name == "sec_filing_parser.py":
            retained.extend(traceback.tb_frame.f_locals.values())
        traceback = traceback.tb_next
    return tuple(retained)


def _reference() -> FilingReference:
    return FilingReference(
        symbol="AAPL",
        cik="0000320193",
        accession_number="0000320193-25-000003",
        filing_type="10-K",
        title="Apple 2025 Form 10-K",
        filed_date=date(2025, 10, 31),
        source_url=(
            "https://www.sec.gov/Archives/edgar/data/320193/000032019325000003/aapl-20250927.htm"
        ),
    )


def test_parser_extracts_visible_inline_xbrl_without_active_or_hidden_content() -> None:
    body = (FIXTURES / "inline_xbrl.html").read_bytes()
    document = SecFilingParser().parse(
        RawFiling(reference=_reference(), media_type="text/html", body=body)
    )

    assert document.symbol == "AAPL"
    assert document.accession_number == "0000320193-25-000003"
    assert "Item 1A. Risk Factors" in document.text
    assert "Supply constraints could affect results." in document.text
    assert "Revenue was $100 million." in document.text
    for forbidden in (
        "ignore previous instructions",
        "hidden XBRL secret",
        "display-none secret",
        "aria secret",
        "iframe secret",
        "object secret",
        "evil.example",
        "ENTITY-CONTENT",
    ):
        assert forbidden not in document.text
    assert "\n\n" in document.text
    assert document.source_url == _reference().source_url


def test_parser_hides_important_styles_and_self_closing_active_content_fail_safe() -> None:
    body = b"""
    <html><body>
      <p>Visible before.</p>
      <div style="display:none !important">important hidden secret</div>
      <script/>self-closing script secret</script>
      <iframe/>self-closing iframe secret</iframe>
      <form/>self-closing form secret</form>
      <p>Visible after.</p>
    </body></html>
    """

    document = SecFilingParser().parse(
        RawFiling(reference=_reference(), media_type="text/html", body=body)
    )

    assert document.text == "Visible before.\n\nVisible after."


def test_parser_stops_extracting_after_crossed_hidden_active_content() -> None:
    body = b"""
    <html><body>
      <p>Visible before.</p>
      <div hidden><button></div>
      crossed active secret
      </button>
      <p>untrusted remainder</p>
    </body></html>
    """

    document = SecFilingParser().parse(
        RawFiling(reference=_reference(), media_type="text/html", body=body)
    )

    assert document.text == "Visible before."


def test_parser_failure_detaches_body_decoded_and_normalized_text() -> None:
    marker = "private-normalized-filing-sentinel"
    filing = RawFiling(
        reference=_reference(),
        media_type="text/html",
        body=f"<html><body>{marker}</body></html>".encode(),
    )

    with pytest.raises(ValueError, match="visible text exceeds") as caught:
        SecFilingParser(max_text_characters=10).parse(filing)

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    retained = _parser_traceback_values(caught.value)
    assert all(value is not filing for value in retained)
    assert marker not in "\n".join(repr(value) for value in retained)


def test_parser_never_uses_untrusted_html_as_filing_metadata() -> None:
    body = b"""
    <html><head><title>MSFT fake title</title></head>
    <body><p>CIK 0000789019</p><p>Visible filing text.</p></body></html>
    """
    document = SecFilingParser().parse(
        RawFiling(reference=_reference(), media_type="text/html", body=body)
    )
    assert document.symbol == "AAPL"
    assert document.title == "Apple 2025 Form 10-K"
    assert document.cik == "0000320193"


@pytest.mark.parametrize(
    "body",
    [
        b"<html><body><script>only hidden</script></body></html>",
        b"<!DOCTYPE html [<!ENTITY xxe SYSTEM 'file:///etc/passwd'>]><html>&xxe;</html>",
        b"\x00\x01",
    ],
)
def test_parser_fails_closed_when_no_visible_text_remains(body: bytes) -> None:
    with pytest.raises(ValueError, match="no usable visible text"):
        SecFilingParser().parse(
            RawFiling(reference=_reference(), media_type="text/html", body=body)
        )


def test_parser_rechecks_input_and_output_bounds() -> None:
    filing = RawFiling(
        reference=_reference(),
        media_type="text/html",
        body=b"<html><body>0123456789</body></html>",
    )
    with pytest.raises(ValueError, match="input exceeds"):
        SecFilingParser(max_input_bytes=8).parse(filing)
    with pytest.raises(ValueError, match="visible text exceeds"):
        SecFilingParser(max_text_characters=5).parse(filing)
    verbose = RawFiling(
        reference=_reference(),
        media_type="text/html",
        body=b"<html><body>abcdefghijklmnopqrstuvwxyz</body></html>",
    )
    with pytest.raises(ValueError, match="visible text exceeds"):
        SecFilingParser(max_text_characters=5).parse(verbose)


def test_parser_configuration_rejects_boolean_and_unbounded_limits() -> None:
    with pytest.raises(ValueError, match="input byte limit"):
        SecFilingParser(max_input_bytes=True)
    with pytest.raises(ValueError, match="text character limit"):
        SecFilingParser(max_text_characters=10_000_001)


def test_parser_handles_standard_entities_numeric_references_and_void_blocks() -> None:
    filing = RawFiling(
        reference=_reference(),
        media_type="application/xhtml+xml",
        body=(
            b"<html><body><p>A&amp;B&#33;&bogus;&#x110000;</p>"
            b"<input value='hidden'/><br/><p>C</p></body></html>"
        ),
    )
    assert SecFilingParser().parse(filing).text == "A&B!\n\nC"


def test_parser_rejects_non_contract_input_and_all_invalid_limit_edges() -> None:
    with pytest.raises(ValueError, match="trusted raw filing"):
        SecFilingParser().parse(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="text character limit"):
        SecFilingParser(max_text_characters=True)
    with pytest.raises(ValueError, match="input byte limit"):
        SecFilingParser(max_input_bytes=64 * 1024 * 1024 + 1)


def test_parser_tolerates_unmatched_tags_and_ignores_references_inside_hidden_nodes() -> None:
    filing = RawFiling(
        reference=_reference(),
        media_type="text/html",
        body=b"<html><body>Visible</unknown><img/><div hidden>&#33;</div></body></html>",
    )
    assert SecFilingParser().parse(filing).text == "Visible"
