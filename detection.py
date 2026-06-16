from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as _np
import fasttext
import fasttext.FastText as _ft_module
from gliner import GLiNER

from config import (
    logger,
    LID_MODEL_PATH, MODEL_THRESHOLD, CHUNK_SIZE, CHUNK_OVERLAP,
    WHITELIST, WHITELIST_RE,
)

# NumPy 2.x compatibility: fasttext calls np.array(..., copy=False) which
# raises ValueError in NumPy 2.x when a copy is unavoidable.
def _ft_predict_fixed(self, text, k=1, threshold=0.0, on_unicode_error="strict"):
    def _check(entry):
        if "\n" in entry:
            raise ValueError("predict processes one line at a time (remove '\\n')")
        return entry + "\n"
    if isinstance(text, list):
        text = [_check(t) for t in text]
        all_labels, all_probs = self.f.multilinePredict(text, k, threshold, on_unicode_error)
        return all_labels, all_probs
    predictions = self.f.predict(_check(text), k, threshold, on_unicode_error)
    if predictions:
        probs, labels = zip(*predictions)
    else:
        probs, labels = [], ()
    return labels, _np.asarray(probs)

_ft_module._FastText.predict = _ft_predict_fixed

logger.info("Loading local language detector")
lid_model = fasttext.load_model(LID_MODEL_PATH)

logger.info("Loading local multilingual PII model")
pii_model = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1", local_files_only=True)
logger.info("Models loaded")


PII_LABELS = [
    "person", "organization", "street address", "location",
    "email address", "phone number", "username", "password",
    "api key", "access token", "secret", "ip address", "mac address",
    "hostname", "domain name", "url", "bank account number", "iban",
    "credit card number", "tax identification number",
    "national identification number", "passport number",
    "driver license number", "health insurance number",
    "customer identifier", "contract number", "serial number",
    "license key", "postal code", "date of birth", "social media handle",
]

LABEL_TO_TOKEN: dict[str, str] = {
    "person": "PERSON", "organization": "ORGANIZATION",
    "street address": "ADDRESS", "location": "LOCATION",
    "email address": "EMAIL", "phone number": "PHONE",
    "username": "USERNAME", "password": "PASSWORD",
    "api key": "API_KEY", "access token": "ACCESS_TOKEN", "secret": "SECRET",
    "ip address": "IP_ADDRESS", "mac address": "MAC_ADDRESS",
    "hostname": "HOSTNAME", "domain name": "DOMAIN", "url": "URL",
    "bank account number": "BANK_ACCOUNT", "iban": "IBAN",
    "credit card number": "CREDIT_CARD",
    "tax identification number": "TAX_ID",
    "national identification number": "NATIONAL_ID",
    "passport number": "PASSPORT", "driver license number": "DRIVER_LICENSE",
    "health insurance number": "HEALTH_ID",
    "customer identifier": "CUSTOMER_ID", "contract number": "CONTRACT_ID",
    "serial number": "SERIAL_NUMBER", "license key": "LICENSE_KEY",
    "postal code": "POSTAL_CODE", "date of birth": "DATE_OF_BIRTH",
    "social media handle": "SOCIAL_HANDLE",
}

# High-confidence, language-independent recognizers. These complement the NER model.
REGEX_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL",       re.compile(r"(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,63}\b", re.I)),
    ("IP_ADDRESS",  re.compile(
        r"(?<![\w.])(?:25[0-5]|2[0-4]\d|1?\d?\d)"
        r"(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?![\w.])"
    )),
    ("IP_ADDRESS",  re.compile(r"(?<![\w:])(?:[A-F0-9]{0,4}:){2,7}[A-F0-9]{0,4}(?![\w:])", re.I)),
    ("MAC_ADDRESS", re.compile(r"\b(?:[0-9A-F]{2}[:-]){5}[0-9A-F]{2}\b", re.I)),
    ("URL",         re.compile(r"\bhttps?://[^\s<>'\"]+", re.I)),
    ("IBAN",        re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]){11,30}\b", re.I)),
    ("CREDIT_CARD", re.compile(r"(?<!\d)(?:\d[ -]*?){13,19}(?!\d)")),
    ("SECRET",      re.compile(
        r"(?i)\b(?:api[_ -]?key|access[_ -]?token|refresh[_ -]?token|password|passwd|secret)"
        r"\s*[:=]\s*['\"]?([A-Za-z0-9_./+=:@~-]{8,})"
    )),
    ("PRIVATE_KEY", re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----.*?"
        r"-----END (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
        re.I | re.S,
    )),
    # Alphanumeric reference codes: 2-4 uppercase letters + 8+ digits (e.g. SC202211296755).
    ("CONTRACT_ID", re.compile(r"\b[A-Z]{2,4}\d{8,}\b")),
]

HOSTNAME_RE = re.compile(
    r"(?<![\w.-])(?=[A-Za-z0-9.-]{4,253}\b)"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"[A-Za-z]{2,63}(?![\w.-])"
)
USER_FIELD_RE = re.compile(
    r"(?im)\b(?:user(?:name)?|login|account)\s*[:=]\s*([^\s,;]{2,128})"
)
# Captures names following German honorifics: "Herr Bethäuser", "Frau Anna Müller"
SALUTATION_RE = re.compile(
    r"(?:Herr(?:n)?|Frau)\s+([A-ZÄÖÜ][a-zäöüß-]{2,}(?:\s+[A-ZÄÖÜ][a-zäöüß-]{2,})?)",
    re.UNICODE,
)


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str
    score: float
    source: str


def detect_language(text: str) -> tuple[str, float]:
    sample = re.sub(r"\s+", " ", text).strip()[:10000]
    if len(sample) < 40:
        return "unknown", 0.0
    labels, probabilities = lid_model.predict(sample, k=1)
    return labels[0].removeprefix("__label__"), float(probabilities[0])


def iter_chunks(text: str) -> Iterable[tuple[int, str]]:
    if len(text) <= CHUNK_SIZE:
        yield 0, text
        return
    start = 0
    while start < len(text):
        end = min(len(text), start + CHUNK_SIZE)
        if end < len(text):
            split = max(
                text.rfind("\n\n", start + CHUNK_SIZE // 2, end),
                text.rfind("\n",   start + CHUNK_SIZE // 2, end),
                text.rfind(". ",   start + CHUNK_SIZE // 2, end),
                text.rfind(" ",    start + CHUNK_SIZE // 2, end),
            )
            if split > start:
                end = split + 1
        yield start, text[start:end]
        if end >= len(text):
            break
        start = max(start + 1, end - CHUNK_OVERLAP)


def regex_spans(text: str) -> list[Span]:
    spans: list[Span] = []
    for label, pattern in REGEX_RULES:
        for match in pattern.finditer(text):
            start, end = match.span(1) if match.lastindex else match.span()
            spans.append(Span(start, end, label, 0.99, "regex"))
    for match in HOSTNAME_RE.finditer(text):
        if "@" not in match.group(0):
            spans.append(Span(match.start(), match.end(), "HOSTNAME", 0.88, "regex"))
    for match in USER_FIELD_RE.finditer(text):
        spans.append(Span(match.start(1), match.end(1), "USERNAME", 0.94, "regex"))
    for match in SALUTATION_RE.finditer(text):
        spans.append(Span(match.start(1), match.end(1), "PERSON", 0.92, "regex"))
    return spans


def model_spans(text: str) -> list[Span]:
    spans: list[Span] = []
    for offset, chunk in iter_chunks(text):
        for entity in pii_model.predict_entities(chunk, PII_LABELS, threshold=MODEL_THRESHOLD, flat_ner=True):
            label = LABEL_TO_TOKEN.get(entity["label"].lower(), "SENSITIVE")
            spans.append(Span(
                offset + int(entity["start"]),
                offset + int(entity["end"]),
                label,
                float(entity.get("score", MODEL_THRESHOLD)),
                "gliner",
            ))
    return spans


def luhn_valid(value: str) -> bool:
    digits = [int(ch) for ch in value if ch.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def normalize_spans(text: str, spans: list[Span]) -> list[Span]:
    filtered: list[Span] = []
    for span in spans:
        value = text[span.start:span.end].strip()
        if not value:
            continue
        if span.label == "CREDIT_CARD" and not luhn_valid(value):
            continue
        if value.casefold() in WHITELIST:
            continue
        if any(p.fullmatch(value) for p in WHITELIST_RE):
            continue
        filtered.append(span)

    # Prefer high-confidence regex, then higher score, then longer spans.
    ordered = sorted(filtered, key=lambda s: (0 if s.source == "regex" else 1, -s.score, -(s.end - s.start), s.start))
    accepted: list[Span] = []
    for candidate in ordered:
        if any(candidate.start < c.end and candidate.end > c.start for c in accepted):
            continue
        accepted.append(candidate)
    return sorted(accepted, key=lambda s: (s.start, s.end))


class Pseudonymizer:
    def __init__(self) -> None:
        self.counters: defaultdict[str, int] = defaultdict(int)
        self.mapping: dict[tuple[str, str], str] = {}

    def token_for(self, label: str, original: str) -> str:
        key = (label, original.casefold())
        if key not in self.mapping:
            self.counters[label] += 1
            self.mapping[key] = f"[{label}_{self.counters[label]:03d}]"
        return self.mapping[key]

    def apply(self, text: str, spans: list[Span]) -> str:
        result = text
        for span in reversed(spans):
            original = text[span.start:span.end]
            token = self.token_for(span.label, original)
            result = result[:span.start] + token + result[span.end:]
        return result


def consistency_sweep(pages: list[str], pseudonymizer: Pseudonymizer) -> list[str]:
    """Second pass: replace remaining occurrences of already-detected entity values.

    The NER model may miss the same name in a different context (e.g. a signature
    block vs. the body). After all pages are processed we know every unique entity
    value; this sweeps each page one more time to catch stragglers.
    """
    entries = sorted(
        [
            (value_folded, token)
            for (_, value_folded), token in pseudonymizer.mapping.items()
            if len(value_folded) >= 4
            and value_folded not in WHITELIST
            and not any(p.fullmatch(value_folded) for p in WHITELIST_RE)
        ],
        key=lambda x: len(x[0]),
        reverse=True,
    )
    if not entries:
        return pages
    result = []
    for page in pages:
        text = page
        for value, token in entries:
            text = re.sub(re.escape(value), token, text, flags=re.IGNORECASE | re.UNICODE)
        result.append(text)
    return result
