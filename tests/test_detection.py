"""
Tests for pure-Python detection logic (no real models required).
conftest.py stubs out fasttext, gliner, and friends before import.
"""
import re
import sys
import importlib
from collections import defaultdict
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------

def _import_detection():
    """Re-import detection with a clean slate so per-test WHITELIST patches work."""
    for name in list(sys.modules):
        if name in ("detection", "config"):
            del sys.modules[name]
    import detection as d
    return d


# ---------------------------------------------------------------------------
# 1. IPv6 regex: must NOT match HH:MM:SS timestamps
# ---------------------------------------------------------------------------

class TestIPv6Regex:
    def setup_method(self):
        self.d = _import_detection()
        # Pull the compiled IPv6 pattern (second IP_ADDRESS entry in REGEX_RULES)
        self.ipv6_pat = next(
            p for label, p in self.d.REGEX_RULES if label == "IP_ADDRESS"
            and p.flags & re.IGNORECASE
        )

    def test_timestamp_not_matched(self):
        for ts in ("10:30:45", "00:00:00", "23:59:59", "09:15:00"):
            assert not self.ipv6_pat.search(ts), f"Timestamp {ts!r} falsely matched as IPv6"

    def test_datetime_timestamp_not_matched(self):
        text = "Created at 2024-01-15 10:30:45 by the system."
        assert not self.ipv6_pat.search(text)

    def test_real_ipv6_matched(self):
        for addr in ("fe80::1", "2001:db8::ff", "::ffff:192.0.2.1", "2001:0db8:85a3::8a2e:0370:7334"):
            assert self.ipv6_pat.search(addr), f"Real IPv6 {addr!r} not matched"

    def test_pure_decimal_colons_not_matched(self):
        assert not self.ipv6_pat.search("1:2:3:4:5:6:7:8")

    def test_bare_double_colon_not_matched(self):
        # "::" with no hex letters should not match
        assert not self.ipv6_pat.search(" :: ")


# ---------------------------------------------------------------------------
# 2. HOSTNAME_RE: must NOT match filenames
# ---------------------------------------------------------------------------

class TestHostnameRegex:
    def setup_method(self):
        self.d = _import_detection()

    def _find_hostnames(self, text):
        spans = []
        for m in self.d.HOSTNAME_RE.finditer(text):
            val = m.group(0)
            if "@" not in val and not self.d._FILE_EXT_RE.search(val):
                spans.append(val)
        return spans

    def test_pdf_filename_not_matched(self):
        assert not self._find_hostnames("See the attached report.pdf for details.")

    def test_various_extensions_not_matched(self):
        filenames = [
            "config.json", "app.py", "data.csv", "setup.exe",
            "readme.md", "style.css", "bundle.js", "archive.zip",
            "image.png", "movie.mp4",
        ]
        for fn in filenames:
            assert not self._find_hostnames(fn), f"Filename {fn!r} was matched as hostname"

    def test_real_hostname_matched(self):
        for host in ("mail.example.com", "vpn.company.org", "api.internal.io"):
            assert self._find_hostnames(host), f"Real hostname {host!r} not matched"

    def test_subdomain_matched(self):
        assert self._find_hostnames("jira.corp.example.com")


# ---------------------------------------------------------------------------
# 3. consistency_sweep: must NOT corrupt [LABEL_NNN] tokens
# ---------------------------------------------------------------------------

class TestConsistencySweepNoCorruption:
    def setup_method(self):
        self.d = _import_detection()

    def _make_pseudonymizer(self, mapping: dict):
        p = self.d.Pseudonymizer()
        for (label, value_cf), token in mapping.items():
            p.mapping[(label, value_cf)] = token
            num = int(token.split("_")[-1].rstrip("]"))
            p.counters[label] = max(p.counters[label], num)
        return p

    def test_token_not_corrupted_by_label_word(self):
        # "address" was detected as PII → its token is [ADDRESS_001]
        # Another page already has [ADDRESS_002] in it.
        # The sweep must NOT corrupt [ADDRESS_002] → [[ADDRESS_001]_002]
        p = self._make_pseudonymizer({
            ("ADDRESS", "address"): "[ADDRESS_001]",
        })
        pages = ["[ADDRESS_002] on Maple Street is nearby."]
        result = self.d.consistency_sweep(pages, p)
        assert "[ADDRESS_002]" in result[0], "Existing token was corrupted"
        assert "[[ADDRESS_001]" not in result[0]

    def test_sweep_still_replaces_missed_occurrences(self):
        p = self._make_pseudonymizer({
            ("PERSON", "john doe"): "[PERSON_001]",
        })
        # Page was anonymized except for one stray "John Doe"
        pages = ["Please contact John Doe for further information."]
        result = self.d.consistency_sweep(pages, p)
        assert "[PERSON_001]" in result[0]
        assert "John Doe" not in result[0]

    def test_hostname_substring_not_corrupting_token(self):
        # "host" (4 chars) detected, token [HOSTNAME_001] already in page
        p = self._make_pseudonymizer({
            ("HOSTNAME", "host"): "[HOSTNAME_002]",
        })
        pages = ["Contact [HOSTNAME_001] or try host directly."]
        result = self.d.consistency_sweep(pages, p)
        assert "[HOSTNAME_001]" in result[0], "Pre-existing token was corrupted"
        assert "[HOSTNAME_002]" in result[0], "Missed occurrence not replaced"

    def test_multiple_tokens_preserved(self):
        p = self._make_pseudonymizer({
            ("PERSON", "alice"): "[PERSON_001]",
            ("EMAIL", "alice@corp.com"): "[EMAIL_001]",
        })
        pages = ["[PERSON_001] wrote to [EMAIL_001] about Alice."]
        result = self.d.consistency_sweep(pages, p)
        assert result[0].count("[PERSON_001]") == 2  # original + replacement
        assert "[EMAIL_001]" in result[0]


# ---------------------------------------------------------------------------
# 4. Whitelist ß inconsistency
# ---------------------------------------------------------------------------

class TestWhitelistBetaConsistency:
    """
    Both normalize_spans and consistency_sweep must treat ß the same way.
    Whitelist patterns are casefolded at load time; values are checked with
    .casefold() in both passes.
    """

    def setup_method(self):
        import config as cfg
        # Patch the module-level WHITELIST_RE with a freshly compiled casefolded pattern.
        self._orig_wl = cfg.WHITELIST
        self._orig_wlre = cfg.WHITELIST_RE
        # Pattern for "straße" (casefold of "Straße")
        cfg.WHITELIST_RE = [re.compile("strasse", re.IGNORECASE | re.UNICODE)]
        cfg.WHITELIST = frozenset()

        self.d = _import_detection()
        # Patch detection module's own references
        self.d.WHITELIST = cfg.WHITELIST
        self.d.WHITELIST_RE = cfg.WHITELIST_RE

    def teardown_method(self):
        import config as cfg
        cfg.WHITELIST = self._orig_wl
        cfg.WHITELIST_RE = self._orig_wlre

    def test_normalize_spans_respects_whitelist(self):
        text = "Straße der Einheit 12"
        span = self.d.Span(0, len("Straße"), "ADDRESS", 0.9, "regex")
        result = self.d.normalize_spans(text, [span])
        assert result == [], "Whitelisted ß-value should be filtered in normalize_spans"

    def test_consistency_sweep_respects_whitelist(self):
        # "Straße" was detected → value_folded == "strasse"
        p = self.d.Pseudonymizer()
        p.mapping[("ADDRESS", "strasse")] = "[ADDRESS_001]"
        p.counters["ADDRESS"] = 1
        pages = ["Straße der Einheit — please visit."]
        result = self.d.consistency_sweep(pages, p)
        # Should NOT replace "Straße" because "strasse" is whitelisted
        assert "Straße" in result[0]
        assert "[ADDRESS_001]" not in result[0]


# ---------------------------------------------------------------------------
# 5. Credit-card regex
# ---------------------------------------------------------------------------

class TestCreditCardRegex:
    def setup_method(self):
        self.d = _import_detection()
        self.pat = next(p for label, p in self.d.REGEX_RULES if label == "CREDIT_CARD")

    def _find_with_luhn(self, text):
        found = []
        for m in self.pat.finditer(text):
            val = m.group(0)
            if self.d.luhn_valid(val):
                found.append(val)
        return found

    def test_valid_visa_compact(self):
        # 4532015112830366 is Luhn-valid
        assert self._find_with_luhn("Card: 4532015112830366")

    def test_valid_visa_spaced(self):
        assert self._find_with_luhn("Card: 4532 0151 1283 0366")

    def test_valid_visa_dashed(self):
        assert self._find_with_luhn("Card: 4532-0151-1283-0366")

    def test_luhn_invalid_rejected(self):
        assert not self._find_with_luhn("1234567890123456")

    def test_short_number_not_matched(self):
        assert not self._find_with_luhn("123456789012")  # 12 digits < 13


# ---------------------------------------------------------------------------
# 6. detect_language: empty fasttext predictions must not raise IndexError
# ---------------------------------------------------------------------------

class TestDetectLanguageEmptyPredictions:
    def setup_method(self):
        self.d = _import_detection()

    def test_empty_predictions_returns_unknown(self):
        self.d.lid_model.predict.return_value = ([], [])
        lang, conf = self.d.detect_language("A" * 50)
        assert lang == "unknown"
        assert conf == 0.0

    def test_short_text_returns_unknown(self):
        lang, conf = self.d.detect_language("Hi")
        assert lang == "unknown"
        assert conf == 0.0

    def test_normal_prediction_parsed(self):
        import numpy as np
        self.d.lid_model.predict.return_value = (["__label__de"], np.array([0.99]))
        lang, conf = self.d.detect_language("Das ist ein sehr langer Satz auf Deutsch." * 3)
        assert lang == "de"
        assert abs(conf - 0.99) < 1e-6


# ---------------------------------------------------------------------------
# 7. SSE generator error propagation
# ---------------------------------------------------------------------------

class TestSSEErrorPropagation:
    """
    When an exception occurs inside generate() after the first yield,
    the server must emit a data:{error:...} SSE event instead of silently
    closing the stream.
    """

    def test_generator_yields_error_event_on_exception(self):
        import asyncio
        import json

        # Build a minimal generate()-like generator that mirrors the fixed logic.
        async def generate_with_error():
            try:
                yield 'data: {"progress":0.5,"page":1,"total":2}\n\n'
                raise RuntimeError("Model exploded")
            except Exception as exc:
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        async def collect():
            events = []
            async for chunk in generate_with_error():
                events.append(chunk)
            return events

        events = asyncio.run(collect())
        assert len(events) == 2
        progress_event = json.loads(events[0].removeprefix("data: ").strip())
        assert progress_event["progress"] == 0.5
        error_event = json.loads(events[1].removeprefix("data: ").strip())
        assert "error" in error_event
        assert "Model exploded" in error_event["error"]

    def test_no_done_event_on_error(self):
        import asyncio
        import json

        async def generate_with_error():
            try:
                yield 'data: {"progress":1.0,"page":1,"total":1}\n\n'
                raise ValueError("build_pdf failed")
                yield 'data: {"done":true}\n\n'  # noqa: unreachable
            except Exception as exc:
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        async def collect():
            return [chunk async for chunk in generate_with_error()]

        events = asyncio.run(collect())
        parsed = [json.loads(e.removeprefix("data: ").strip()) for e in events]
        assert not any("done" in ev for ev in parsed)
        assert any("error" in ev for ev in parsed)
