from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from html.entities import html5
from html.parser import HTMLParser

from app.research.domain import FilingDocument, RawFiling

_SKIPPED_TAGS = frozenset(
    {
        "head",
        "script",
        "style",
        "template",
        "form",
        "iframe",
        "object",
        "applet",
        "canvas",
        "button",
        "input",
        "select",
        "textarea",
        "noscript",
        "svg",
        "math",
        "ix:hidden",
    }
)
_BLOCK_TAGS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "div",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "p",
        "section",
        "table",
        "tr",
    }
)
_VOID_TAGS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "source",
        "track",
        "wbr",
    }
)
_HIDDEN_STYLE = re.compile(
    r"(?:^|;)\s*(?:display\s*:\s*none|visibility\s*:\s*hidden)"
    r"\s*(?:!\s*important\s*)?(?:;|$)",
    re.I,
)
_ACTIVE_CONTAINER_TAGS = frozenset(
    {
        "applet",
        "button",
        "canvas",
        "form",
        "iframe",
        "noscript",
        "object",
        "script",
        "select",
        "style",
        "template",
        "textarea",
    }
)
_INTERNAL_DTD = re.compile(r"<!DOCTYPE[^[]*\[[\s\S]*?\]>", re.I)
_ENTITY_DECLARATION = re.compile(r"<!ENTITY[\s\S]*?>", re.I)


class _TextLimitError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class _ParseOutcome:
    document: FilingDocument | None = None
    error: str | None = None


def _require_document(outcome: _ParseOutcome) -> FilingDocument:
    if outcome.document is None:
        raise ValueError(
            outcome.error or "filing parser could not safely process content"
        ) from None
    return outcome.document


class _VisibleTextParser(HTMLParser):
    def __init__(self, maximum: int) -> None:
        super().__init__(convert_charrefs=False)
        self._maximum = maximum
        self._parts: list[str] = []
        self._characters = 0
        self._stack: list[tuple[str, bool]] = []
        self._skip_depth = 0
        self._failed_closed = False

    @property
    def text(self) -> str:
        return "".join(self._parts)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = tag.lower()
        hidden = self._skip_depth > 0 or self._is_hidden(normalized, attrs)
        if normalized not in _VOID_TAGS:
            self._stack.append((normalized, hidden))
            if hidden:
                self._skip_depth += 1
        if not hidden and normalized in _BLOCK_TAGS:
            self._append("\n\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = tag.lower()
        if normalized in _ACTIVE_CONTAINER_TAGS:
            hidden = self._skip_depth > 0 or self._is_hidden(normalized, attrs)
            self._stack.append((normalized, hidden))
            if hidden:
                self._skip_depth += 1
            return
        if (
            self._skip_depth == 0
            and not self._is_hidden(normalized, attrs)
            and normalized in _BLOCK_TAGS
        ):
            self._append("\n\n")

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        closing_hidden = self._skip_depth > 0
        index = next(
            (
                index
                for index in range(len(self._stack) - 1, -1, -1)
                if self._stack[index][0] == normalized
            ),
            None,
        )
        if index is not None:
            if any(
                tag in _ACTIVE_CONTAINER_TAGS and hidden for tag, hidden in self._stack[index + 1 :]
            ):
                self._failed_closed = True
            removed = self._stack[index:]
            self._stack = self._stack[:index]
            self._skip_depth -= sum(1 for _, hidden in removed if hidden)
        if not closing_hidden and normalized in _BLOCK_TAGS:
            self._append("\n\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._append(data)

    def handle_entityref(self, name: str) -> None:
        if self._skip_depth == 0 and (name in html5 or f"{name};" in html5):
            self._append(html.unescape(f"&{name};"))

    def handle_charref(self, name: str) -> None:
        if self._skip_depth > 0:
            return
        try:
            number = int(name[1:], 16) if name.lower().startswith("x") else int(name)
            value = chr(number)
        except (ValueError, OverflowError):
            return
        if not _control_like(value):
            self._append(value)

    def handle_decl(self, decl: str) -> None:
        del decl

    def unknown_decl(self, data: str) -> None:
        del data

    def handle_pi(self, data: str) -> None:
        del data

    def _append(self, value: str) -> None:
        if self._failed_closed:
            return
        safe = "".join(" " if _control_like(character) else character for character in value)
        self._characters += len(safe)
        if self._characters > self._maximum * 4:
            raise _TextLimitError
        self._parts.append(safe)

    @staticmethod
    def _is_hidden(tag: str, attrs: list[tuple[str, str | None]]) -> bool:
        if tag in _SKIPPED_TAGS:
            return True
        values = {name.lower(): (value or "") for name, value in attrs}
        return (
            "hidden" in values
            or values.get("aria-hidden", "").strip().lower() == "true"
            or bool(_HIDDEN_STYLE.search(values.get("style", "")))
        )


def _control_like(character: str) -> bool:
    return unicodedata.category(character) in {"Cc", "Cf"} and character not in "\n\t"


class SecFilingParser:
    """Network-free visible-text parser for SEC HTML and Inline XBRL filings."""

    def __init__(
        self,
        *,
        max_input_bytes: int = 16 * 1024 * 1024,
        max_text_characters: int = 5_000_000,
    ) -> None:
        if type(max_input_bytes) is not int or not 1 <= max_input_bytes <= 64 * 1024 * 1024:
            raise ValueError("filing parser input byte limit is invalid")
        if type(max_text_characters) is not int or not 1 <= max_text_characters <= 5_000_000:
            raise ValueError("filing parser text character limit is invalid")
        self._max_input = max_input_bytes
        self._max_text = max_text_characters

    def parse(self, filing: RawFiling) -> FilingDocument:
        outcome = self._parse_outcome(filing)
        del filing
        return _require_document(outcome)

    def _parse_outcome(self, filing: object) -> _ParseOutcome:
        if not isinstance(filing, RawFiling):
            return _ParseOutcome(error="filing parser requires a trusted raw filing")
        if len(filing.body) > self._max_input:
            return _ParseOutcome(error="filing parser input exceeds the configured limit")
        decoded = filing.body.decode("utf-8-sig", errors="replace")
        decoded = _INTERNAL_DTD.sub("", decoded)
        decoded = _ENTITY_DECLARATION.sub("", decoded)
        parser = _VisibleTextParser(self._max_text)
        try:
            parser.feed(decoded)
            parser.close()
        except _TextLimitError:
            return _ParseOutcome(error="filing visible text exceeds the configured limit")
        except Exception:
            return _ParseOutcome(error="filing parser could not safely process content")
        normalized = self._normalize(parser.text)
        if not normalized:
            return _ParseOutcome(error="filing parser produced no usable visible text")
        if len(normalized) > self._max_text:
            return _ParseOutcome(error="filing visible text exceeds the configured limit")
        reference = filing.reference
        try:
            document = FilingDocument(
                symbol=reference.symbol,
                cik=reference.cik,
                accession_number=reference.accession_number,
                filing_type=reference.filing_type,
                title=reference.title,
                filed_date=reference.filed_date,
                source_url=reference.source_url,
                text=normalized,
            )
        except (TypeError, ValueError):
            return _ParseOutcome(error="filing parser produced an invalid document")
        return _ParseOutcome(document=document)

    @staticmethod
    def _normalize(value: str) -> str:
        value = value.replace("\r\n", "\n").replace("\r", "\n")
        value = re.sub(r"[^\S\n]+", " ", value)
        value = re.sub(r" *\n *", "\n", value)
        value = re.sub(r"\n{3,}", "\n\n", value)
        return value.strip()


__all__ = ["SecFilingParser"]
