from __future__ import annotations

import asyncio
import base64
import hashlib
import html
import io
import json
import logging
import os
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as _np
import fasttext
import fasttext.FastText as _ft_module
import pymupdf

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

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from gliner import GLiNER
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("anonimabobulator")
logging.getLogger("gliner").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

MAX_UPLOAD_BYTES = int(os.getenv("APP_MAX_UPLOAD_MB", "30")) * 1024 * 1024
LID_MODEL_PATH = os.getenv("APP_LID_MODEL", "/opt/models/lid.176.ftz")
MIN_TEXT_CHARS = 80
MODEL_THRESHOLD = float(os.getenv("APP_MODEL_THRESHOLD", "0.45"))
CHUNK_SIZE = int(os.getenv("APP_CHUNK_SIZE", "1800"))
CHUNK_OVERLAP = int(os.getenv("APP_CHUNK_OVERLAP", "180"))


def _load_whitelists() -> tuple[frozenset[str], list[re.Pattern[str]]]:
    terms: set[str] = set()
    patterns: list[re.Pattern[str]] = []
    paths = sorted(Path("/app").glob("whitelist-*.txt"))
    for path in paths:
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("~"):
                    try:
                        patterns.append(re.compile(line[1:], re.IGNORECASE | re.UNICODE))
                    except re.error as exc:
                        logger.warning("Whitelist regex error in %s: %r — %s", path.name, line, exc)
                else:
                    terms.add(line.casefold())
        except OSError:
            pass
    if terms or patterns:
        logger.info(
            "Whitelist: %d exact term(s) and %d pattern(s) from %d file(s)",
            len(terms), len(patterns), len(paths),
        )
    return frozenset(terms), patterns


WHITELIST, WHITELIST_RE = _load_whitelists()

PII_LABELS = [
    "person",
    "organization",
    "street address",
    "location",
    "email address",
    "phone number",
    "username",
    "password",
    "api key",
    "access token",
    "secret",
    "ip address",
    "mac address",
    "hostname",
    "domain name",
    "url",
    "bank account number",
    "iban",
    "credit card number",
    "tax identification number",
    "national identification number",
    "passport number",
    "driver license number",
    "health insurance number",
    "customer identifier",
    "contract number",
    "serial number",
    "license key",
    "postal code",
    "date of birth",
    "social media handle",
]

LABEL_TO_TOKEN = {
    "person": "PERSON",
    "organization": "ORGANIZATION",
    "street address": "ADDRESS",
    "location": "LOCATION",
    "email address": "EMAIL",
    "phone number": "PHONE",
    "username": "USERNAME",
    "password": "PASSWORD",
    "api key": "API_KEY",
    "access token": "ACCESS_TOKEN",
    "secret": "SECRET",
    "ip address": "IP_ADDRESS",
    "mac address": "MAC_ADDRESS",
    "hostname": "HOSTNAME",
    "domain name": "DOMAIN",
    "url": "URL",
    "bank account number": "BANK_ACCOUNT",
    "iban": "IBAN",
    "credit card number": "CREDIT_CARD",
    "tax identification number": "TAX_ID",
    "national identification number": "NATIONAL_ID",
    "passport number": "PASSPORT",
    "driver license number": "DRIVER_LICENSE",
    "health insurance number": "HEALTH_ID",
    "customer identifier": "CUSTOMER_ID",
    "contract number": "CONTRACT_ID",
    "serial number": "SERIAL_NUMBER",
    "license key": "LICENSE_KEY",
    "postal code": "POSTAL_CODE",
    "date of birth": "DATE_OF_BIRTH",
    "social media handle": "SOCIAL_HANDLE",
}

# High-confidence, language-independent recognizers. These complement the NER model.
REGEX_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,63}\b", re.I)),
    ("IP_ADDRESS", re.compile(
        r"(?<![\w.])(?:25[0-5]|2[0-4]\d|1?\d?\d)"
        r"(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?![\w.])"
    )),
    ("IP_ADDRESS", re.compile(r"(?<![\w:])(?:[A-F0-9]{0,4}:){2,7}[A-F0-9]{0,4}(?![\w:])", re.I)),
    ("MAC_ADDRESS", re.compile(r"\b(?:[0-9A-F]{2}[:-]){5}[0-9A-F]{2}\b", re.I)),
    ("URL", re.compile(r"\bhttps?://[^\s<>'\"]+", re.I)),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]){11,30}\b", re.I)),
    ("CREDIT_CARD", re.compile(r"(?<!\d)(?:\d[ -]*?){13,19}(?!\d)")),
    ("SECRET", re.compile(
        r"(?i)\b(?:api[_ -]?key|access[_ -]?token|refresh[_ -]?token|password|passwd|secret)"
        r"\s*[:=]\s*['\"]?([A-Za-z0-9_./+=:@~-]{8,})"
    )),
    ("PRIVATE_KEY", re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----.*?"
        r"-----END (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
        re.I | re.S,
    )),
    # Alphanumeric reference codes: 2-4 uppercase letters followed by 8+ digits.
    # Catches contract IDs like SC202211296755, ticket refs, serial numbers, etc.
    ("CONTRACT_ID", re.compile(r"\b[A-Z]{2,4}\d{8,}\b")),
]

# Common technical identifiers in support tickets.
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


app = FastAPI(
    title="Anonimabobulator",
    docs_url=None,
    redoc_url=None,
)

logger.info("Loading local language detector")
lid_model = fasttext.load_model(LID_MODEL_PATH)

logger.info("Loading local multilingual PII model")
pii_model = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1", local_files_only=True)
logger.info("Models loaded")


HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Anonimabobulator</title>
  <style>
    :root {
      color-scheme: light dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      --panel: color-mix(in srgb, Canvas 94%, #5873ff 6%);
      --border: color-mix(in srgb, CanvasText 20%, transparent);
      --accent: #5873ff;
      --danger: #d84a4a;
      --ok: #2d9f69;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background:
        radial-gradient(circle at 18% 15%, rgba(88,115,255,.17), transparent 32rem),
        Canvas;
      color: CanvasText;
    }
    main { width: min(760px, calc(100% - 32px)); }
    .card {
      padding: clamp(24px, 5vw, 48px);
      border: 1px solid var(--border);
      border-radius: 24px;
      background: var(--panel);
      box-shadow: 0 24px 80px rgba(0,0,0,.12);
    }
    h1 { margin: 0 0 10px; font-size: clamp(28px, 5vw, 44px); letter-spacing: -.04em; }
    p { line-height: 1.55; opacity: .82; }
    #dropzone {
      margin-top: 28px;
      min-height: 250px;
      display: grid;
      place-items: center;
      text-align: center;
      padding: 28px;
      border: 2px dashed var(--border);
      border-radius: 20px;
      cursor: pointer;
      transition: .18s ease;
      background: color-mix(in srgb, Canvas 97%, var(--accent) 3%);
    }
    #dropzone:hover, #dropzone.dragging {
      border-color: var(--accent);
      transform: translateY(-2px);
      background: color-mix(in srgb, Canvas 92%, var(--accent) 8%);
    }
    .icon { font-size: 56px; margin-bottom: 8px; }
    strong { display: block; font-size: 20px; }
    small { opacity: .65; }
    input { display: none; }
    #status { min-height: 28px; margin-top: 20px; font-weight: 650; }
    #status.error { color: var(--danger); }
    #status.ok { color: var(--ok); }
    #progress-wrap { display: none; margin-top: 18px; }
    .busy #progress-wrap { display: block; }
    .busy #dropzone { opacity: .55; pointer-events: none; }
    progress {
      width: 100%; height: 8px;
      border: none; border-radius: 4px;
      background: var(--border);
    }
    progress::-webkit-progress-bar { background: var(--border); border-radius: 4px; }
    progress::-webkit-progress-value {
      background: var(--accent); border-radius: 4px;
      transition: width .25s ease;
    }
    progress::-moz-progress-bar { background: var(--accent); border-radius: 4px; }
    #progress-label {
      display: block; margin-top: 6px;
      font-size: 13px; opacity: .68; text-align: center;
    }
    footer { margin-top: 18px; text-align: center; font-size: 13px; opacity: .62; }
  </style>
</head>
<body>
<main>
  <section class="card" id="card">
    <h1>Anonimabobulator</h1>
    <p>Drop a PDF to anonymize it. Processing stays inside this container. The anonymized PDF downloads automatically.</p>

    <label id="dropzone" for="file">
      <div>
        <div class="icon">⇩</div>
        <strong>Drop the PDF here</strong>
        <p>or click to choose a file</p>
        <small>Maximum size: __MAX_MB__ MB</small>
      </div>
    </label>
    <input id="file" type="file" accept="application/pdf,.pdf">
    <div id="progress-wrap">
      <progress id="bar" value="0" max="1"></progress>
      <span id="progress-label"></span>
    </div>
    <div id="status" role="status" aria-live="polite"></div>
  </section>
  <footer>The original PDF and extracted text are not written to disk.</footer>
</main>

<script>
const card  = document.getElementById("card");
const zone  = document.getElementById("dropzone");
const input = document.getElementById("file");
const status = document.getElementById("status");
const bar   = document.getElementById("bar");
const progressLabel = document.getElementById("progress-label");

function setStatus(msg, kind = "") { status.textContent = msg; status.className = kind; }
function setProgress(value, page, total) {
  bar.value = value;
  progressLabel.textContent = `Page ${page} of ${total}`;
}
function resetProgress() { bar.value = 0; progressLabel.textContent = ""; }

["dragenter", "dragover"].forEach(n => zone.addEventListener(n, e => {
  e.preventDefault(); zone.classList.add("dragging");
}));
["dragleave", "drop"].forEach(n => zone.addEventListener(n, e => {
  e.preventDefault(); zone.classList.remove("dragging");
}));
zone.addEventListener("drop", e => { const f = e.dataTransfer.files[0]; if (f) upload(f); });
input.addEventListener("change", () => { const f = input.files[0]; if (f) upload(f); });

async function upload(file) {
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    setStatus("Choose a PDF file.", "error");
    return;
  }

  card.classList.add("busy");
  setStatus("Uploading…");
  resetProgress();

  const data = new FormData();
  data.append("file", file);

  try {
    const response = await fetch("/anonymize", { method: "POST", body: data });

    if (!response.ok) {
      let message = "The PDF could not be processed.";
      try { message = (await response.json()).detail || message; } catch (_) {}
      throw new Error(message);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    setStatus("Anonymizing…");

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        let event;
        try { event = JSON.parse(line.slice(6)); } catch (_) { continue; }

        if (event.error) throw new Error(event.error);

        if (event.progress != null) setProgress(event.progress, event.page, event.total);

        if (event.done) {
          setProgress(1, event.total, event.total);
          const bytes = Uint8Array.from(atob(event.data), c => c.charCodeAt(0));
          const blob = new Blob([bytes], { type: "application/pdf" });
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url; a.download = event.filename;
          document.body.appendChild(a); a.click(); a.remove();
          setTimeout(() => URL.revokeObjectURL(url), 1000);
          setStatus("Done. The anonymized PDF was downloaded.", "ok");
        }
      }
    }
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    card.classList.remove("busy");
    input.value = "";
  }
}
</script>
</body>
</html>""".replace("__MAX_MB__", str(MAX_UPLOAD_BYTES // 1024 // 1024))


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return HTML_PAGE


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def clean_filename(filename: str) -> str:
    base = Path(filename or "ticket.pdf").stem
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    return (cleaned or "ticket")[:100]


async def read_limited(upload: UploadFile) -> bytes:
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"PDF exceeds the {MAX_UPLOAD_BYTES // 1024 // 1024} MB limit.")
    return data


def extract_pages(pdf_bytes: bytes) -> list[str]:
    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise HTTPException(422, "Invalid or damaged PDF.") from exc

    if document.needs_pass:
        document.close()
        raise HTTPException(422, "Password-protected PDFs are not supported.")

    if document.page_count == 0:
        document.close()
        raise HTTPException(422, "The PDF contains no pages.")

    pages: list[str] = []
    try:
        for page in document:
            pages.append(page.get_text("text", sort=True).strip())
    finally:
        document.close()

    meaningful = sum(ch.isalnum() for text in pages for ch in text)
    if meaningful < MIN_TEXT_CHARS:
        raise HTTPException(
            422,
            "The PDF has no usable text layer. Make sure the PDF was exported using Print to PDF."
        )
    return pages


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
            candidates = [
                text.rfind("\n\n", start + CHUNK_SIZE // 2, end),
                text.rfind("\n", start + CHUNK_SIZE // 2, end),
                text.rfind(". ", start + CHUNK_SIZE // 2, end),
                text.rfind(" ", start + CHUNK_SIZE // 2, end),
            ]
            split = max(candidates)
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
        entities = pii_model.predict_entities(
            chunk,
            PII_LABELS,
            threshold=MODEL_THRESHOLD,
            flat_ner=True,
        )
        for entity in entities:
            label = LABEL_TO_TOKEN.get(entity["label"].lower(), "SENSITIVE")
            spans.append(
                Span(
                    offset + int(entity["start"]),
                    offset + int(entity["end"]),
                    label,
                    float(entity.get("score", MODEL_THRESHOLD)),
                    "gliner",
                )
            )

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
    ordered = sorted(
        filtered,
        key=lambda s: (
            0 if s.source == "regex" else 1,
            -s.score,
            -(s.end - s.start),
            s.start,
        ),
    )

    accepted: list[Span] = []
    for candidate in ordered:
        if any(candidate.start < current.end and candidate.end > current.start for current in accepted):
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


def _consistency_sweep(pages: list[str], pseudonymizer: Pseudonymizer) -> list[str]:
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
        reverse=True,  # longest first so sub-strings don't clobber longer matches
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


def build_pdf(pages: list[str], metadata: list[dict[str, object]]) -> bytes:
    output = io.BytesIO()
    pdfmetrics.registerFont(TTFont("DejaVu", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))

    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="Anonymized Ticket — Anonimabobulator",
        author="Anonimabobulator",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TicketTitle",
        parent=styles["Heading1"],
        fontName="DejaVu-Bold",
        fontSize=15,
        leading=19,
        alignment=TA_LEFT,
        spaceAfter=8,
    )
    info_style = ParagraphStyle(
        "Info",
        parent=styles["Normal"],
        fontName="DejaVu",
        fontSize=8,
        leading=11,
        textColor="#555555",
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "TicketBody",
        parent=styles["Code"],
        fontName="DejaVu",
        fontSize=8.5,
        leading=11.5,
        leftIndent=0,
        rightIndent=0,
        wordWrap="CJK",
    )

    story = []
    for index, text in enumerate(pages):
        data = metadata[index]
        story.append(Paragraph(f"Anonimabobulator · Page {index + 1}", title_style))
        story.append(Paragraph(
            f"Detected language: {html.escape(str(data['language']))} "
            f"({float(data['language_confidence']):.1%}) · "
            f"Replacements: {int(data['entities'])}",
            info_style,
        ))
        escaped = html.escape(text).replace("\n", "<br/>")
        story.append(Paragraph(escaped or " ", body_style))
        if index < len(pages) - 1:
            story.append(PageBreak())

    document.build(story)
    return output.getvalue()


@app.post("/anonymize")
async def anonymize(file: UploadFile = File(...)) -> StreamingResponse:
    filename = file.filename or "ticket.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(415, "Only PDF files are accepted.")

    pdf_bytes = await read_limited(file)
    if not pdf_bytes.startswith(b"%PDF-"):
        raise HTTPException(415, "The uploaded file is not a valid PDF.")

    # Log only a one-way fingerprint and byte size, never content or original filename.
    fingerprint = hashlib.sha256(pdf_bytes).hexdigest()[:12]
    logger.info("Processing PDF fingerprint=%s bytes=%d", fingerprint, len(pdf_bytes))

    # Extract pages before the stream starts so validation errors still produce HTTP errors.
    pages = await asyncio.to_thread(extract_pages, pdf_bytes)
    output_name = f"{clean_filename(filename)}_anonymized.pdf"

    async def generate():
        total = len(pages)
        pseudonymizer = Pseudonymizer()
        output_pages: list[str] = []
        page_metadata: list[dict[str, object]] = []

        for i, text in enumerate(pages):
            language, confidence = await asyncio.to_thread(detect_language, text)
            r_spans = await asyncio.to_thread(regex_spans, text)
            m_spans = await asyncio.to_thread(model_spans, text)
            spans = normalize_spans(text, r_spans + m_spans)
            output_pages.append(pseudonymizer.apply(text, spans))
            page_metadata.append({
                "page": i + 1,
                "language": language,
                "language_confidence": round(confidence, 3),
                "entities": len(spans),
            })
            yield f'data: {{"progress":{round((i + 1) / total, 3)},"page":{i + 1},"total":{total}}}\n\n'

        output_pages = await asyncio.to_thread(_consistency_sweep, output_pages, pseudonymizer)
        result = await asyncio.to_thread(build_pdf, output_pages, page_metadata)

        logger.info(
            "Finished PDF fingerprint=%s pages=%d replacements=%d",
            fingerprint, total,
            sum(int(item["entities"]) for item in page_metadata),
        )

        b64 = base64.b64encode(result).decode()
        yield f'data: {{"done":true,"filename":{json.dumps(output_name)},"data":"{b64}"}}\n\n'

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store, max-age=0",
            "X-Accel-Buffering": "no",
        },
    )
