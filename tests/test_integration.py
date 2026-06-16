"""
Integration tests: real pymupdf + reportlab, no ML models needed.

Strategy
--------
conftest.py installs stubs for pymupdf/reportlab before any test module is
imported. We pop those stubs temporarily to capture the real module objects at
collection time, then restore the stubs so unit tests are unaffected.

Font-dependent tests (build_pdf) are skipped automatically when the
container-specific DejaVu font paths aren't present on the host.
"""
from __future__ import annotations

import importlib
import io
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Capture real modules before / around the stubs
# ---------------------------------------------------------------------------

def _pop_and_import(name: str):
    """Temporarily remove any stub for *name*, import the real thing, restore."""
    stub = sys.modules.pop(name, None)
    try:
        return importlib.import_module(name)
    finally:
        if stub is not None:
            sys.modules[name] = stub


_real_pymupdf = _pop_and_import("pymupdf")

# Clear all reportlab stubs at once (the top-level MagicMock blocks submodule imports),
# then import the real package so it registers all its submodules in sys.modules.
_rl_stubs = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "reportlab" or k.startswith("reportlab.")}
try:
    import reportlab  # noqa: F401  — registers the real package tree
finally:
    for _k, _v in _rl_stubs.items():
        sys.modules[_k] = _v


_DEJAVU = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_fonts_available = Path(_DEJAVU).exists()
needs_fonts = pytest.mark.skipif(
    not _fonts_available,
    reason="DejaVu fonts only present inside the container",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_PADDING = " Padding filler text to satisfy the minimum character threshold required by extract pages. "

def _make_fixture_pdf(
    text: str,
    page_width: float = 595.0,
    page_height: float = 842.0,
) -> bytes:
    """Build a single-page PDF with *text* using pymupdf (no reportlab needed)."""
    # Ensure we always exceed MIN_TEXT_CHARS (80) so extract_pages doesn't reject the fixture.
    padded = text
    while sum(ch.isalnum() for ch in padded) < 90:
        padded += _PADDING
    doc = _real_pymupdf.open()
    page = doc.new_page(width=page_width, height=page_height)
    margin = 36.0
    rect = _real_pymupdf.Rect(margin, margin, page_width - margin, page_height - margin)
    page.insert_textbox(rect, padded, fontsize=12)
    data = doc.tobytes()
    doc.close()
    return data


def _load_pdf_module():
    """Return pdf.py backed by the real pymupdf/reportlab (restores stubs after)."""
    saved = {k: sys.modules.pop(k) for k in list(sys.modules)
             if k in ("pdf", "config") or k.startswith(("pymupdf", "reportlab"))}
    sys.modules["pymupdf"] = _real_pymupdf
    try:
        if "config" in saved:
            sys.modules["config"] = saved.pop("config")
        import pdf as _pdf
        return _pdf
    finally:
        for k, v in saved.items():
            sys.modules[k] = v


# ---------------------------------------------------------------------------
# Tests: extract_pages
# ---------------------------------------------------------------------------

class TestExtractPages:
    def setup_method(self):
        self.pdf = _load_pdf_module()

    def test_extracts_text_from_simple_pdf(self):
        pdf_bytes = _make_fixture_pdf("The quick brown fox jumps over the lazy dog. " * 5)
        pages, sizes = self.pdf.extract_pages(pdf_bytes)
        assert len(pages) == 1
        assert "quick brown fox" in pages[0]

    def test_returns_page_size(self):
        pdf_bytes = _make_fixture_pdf("Some text here.", 612.0, 792.0)
        _, sizes = self.pdf.extract_pages(pdf_bytes)
        assert len(sizes) == 1
        w, h = sizes[0]
        assert abs(w - 612.0) < 2.0, f"Expected width ~612, got {w}"
        assert abs(h - 792.0) < 2.0, f"Expected height ~792, got {h}"

    def test_a4_page_size_preserved(self):
        pdf_bytes = _make_fixture_pdf("Content.", 595.28, 841.89)
        _, sizes = self.pdf.extract_pages(pdf_bytes)
        w, h = sizes[0]
        assert abs(w - 595.28) < 2.0
        assert abs(h - 841.89) < 2.0

    def test_rejects_non_pdf_bytes(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            self.pdf.extract_pages(b"not a pdf")
        assert exc_info.value.status_code == 422

    def test_rejects_empty_text_pdf(self):
        from fastapi import HTTPException
        # A valid PDF with no extractable alphanumeric text
        doc = _real_pymupdf.open()
        doc.new_page()          # blank page, no text inserted
        pdf_bytes = doc.tobytes()
        doc.close()
        with pytest.raises(HTTPException) as exc_info:
            self.pdf.extract_pages(pdf_bytes)
        assert exc_info.value.status_code == 422


# ---------------------------------------------------------------------------
# Tests: build_pdf  (require DejaVu fonts — Linux / container only)
# ---------------------------------------------------------------------------

class TestBuildPdf:
    def setup_method(self):
        self.pdf = _load_pdf_module()

    @needs_fonts
    def test_output_is_valid_pdf(self):
        result = self.pdf.build_pdf(
            ["Hello world. Anonymized content."],
            [{"language": "en", "language_confidence": 0.99, "entities": 2}],
        )
        assert result.startswith(b"%PDF-")

    @needs_fonts
    def test_custom_page_size_embedded(self):
        result = self.pdf.build_pdf(
            ["Some anonymized ticket text."],
            [{"language": "en", "language_confidence": 0.9, "entities": 0}],
            page_size=(612.0, 792.0),
        )
        doc = _real_pymupdf.open(stream=result, filetype="pdf")
        w, h = doc[0].rect.width, doc[0].rect.height
        doc.close()
        assert abs(w - 612.0) < 2.0
        assert abs(h - 792.0) < 2.0

    @needs_fonts
    def test_a4_default_when_no_size_given(self):
        result = self.pdf.build_pdf(
            ["Default page size test."],
            [{"language": "de", "language_confidence": 0.8, "entities": 1}],
        )
        doc = _real_pymupdf.open(stream=result, filetype="pdf")
        w = doc[0].rect.width
        doc.close()
        assert abs(w - 595.28) < 2.0

    @needs_fonts
    def test_multipage_output(self):
        result = self.pdf.build_pdf(
            ["Page one.", "Page two."],
            [
                {"language": "en", "language_confidence": 0.95, "entities": 1},
                {"language": "en", "language_confidence": 0.95, "entities": 0},
            ],
        )
        doc = _real_pymupdf.open(stream=result, filetype="pdf")
        count = doc.page_count
        doc.close()
        assert count == 2


# ---------------------------------------------------------------------------
# Tests: round-trip (extract → build, page size preserved)
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def setup_method(self):
        self.pdf = _load_pdf_module()

    def test_page_size_recorded(self):
        source = _make_fixture_pdf("Ticket text here.", 612.0, 792.0)
        pages, sizes = self.pdf.extract_pages(source)
        assert abs(sizes[0][0] - 612.0) < 2.0
        assert abs(sizes[0][1] - 792.0) < 2.0

    def test_text_extracted_for_anonymization(self):
        source = _make_fixture_pdf("John Doe, john@example.com, +1-555-0100")
        pages, _ = self.pdf.extract_pages(source)
        assert "John Doe" in pages[0] or "john" in pages[0].lower()

    @needs_fonts
    def test_page_size_survives_full_round_trip(self):
        source = _make_fixture_pdf(
            "Confidential ticket content for anonymization.",
            612.0, 792.0,
        )
        pages, sizes = self.pdf.extract_pages(source)
        result = self.pdf.build_pdf(
            pages,
            [{"language": "en", "language_confidence": 0.99, "entities": 0}],
            page_size=sizes[0],
        )
        doc = _real_pymupdf.open(stream=result, filetype="pdf")
        w, h = doc[0].rect.width, doc[0].rect.height
        doc.close()
        assert abs(w - 612.0) < 2.0
        assert abs(h - 792.0) < 2.0
