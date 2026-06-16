from __future__ import annotations

import asyncio
import hashlib
import json
import re
import secrets
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from config import logger, MAX_UPLOAD_BYTES
from detection import Pseudonymizer, regex_spans, model_spans, normalize_spans, consistency_sweep, detect_language
from pdf import extract_pages, build_pdf

app = FastAPI(title="Anonimabobulator", docs_url=None, redoc_url=None)

# Serialises NER + PDF-build so concurrent uploads don't pile up 30 MB each.
_NER_SEM = asyncio.Semaphore(1)

# Short-lived download store: token → (pdf_bytes, filename, expires_at).
_DOWNLOAD_STORE: dict[str, tuple[bytes, str, float]] = {}
_TOKEN_TTL = 300  # seconds


def _new_download_token(pdf_bytes: bytes, filename: str) -> str:
    now = time.monotonic()
    token = secrets.token_urlsafe(24)
    _DOWNLOAD_STORE[token] = (pdf_bytes, filename, now + _TOKEN_TTL)
    expired = [k for k, (_, _, exp) in _DOWNLOAD_STORE.items() if exp < now]
    for k in expired:
        del _DOWNLOAD_STORE[k]
    return token


_CSP = (
    "default-src 'self'; "
    "script-src 'unsafe-inline'; "
    "style-src 'unsafe-inline'; "
    "img-src 'self' data:; "
    "connect-src 'self'; "
    "frame-ancestors 'none';"
)

@app.middleware("http")
async def security_headers(request: Request, call_next) -> Response:
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = _CSP
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response

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
      margin: 0; min-height: 100vh; display: grid; place-items: center;
      background: radial-gradient(circle at 18% 15%, rgba(88,115,255,.17), transparent 32rem), Canvas;
      color: CanvasText;
    }
    main { width: min(760px, calc(100% - 32px)); }
    .card {
      padding: clamp(24px, 5vw, 48px); border: 1px solid var(--border);
      border-radius: 24px; background: var(--panel); box-shadow: 0 24px 80px rgba(0,0,0,.12);
    }
    h1 { margin: 0 0 10px; font-size: clamp(28px, 5vw, 44px); letter-spacing: -.04em; }
    p { line-height: 1.55; opacity: .82; }
    #dropzone {
      margin-top: 28px; min-height: 250px; display: grid; place-items: center;
      text-align: center; padding: 28px; border: 2px dashed var(--border);
      border-radius: 20px; cursor: pointer; transition: .18s ease;
      background: color-mix(in srgb, Canvas 97%, var(--accent) 3%);
    }
    #dropzone:hover, #dropzone.dragging {
      border-color: var(--accent); transform: translateY(-2px);
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
    progress { width: 100%; height: 8px; border: none; border-radius: 4px; background: var(--border); }
    progress::-webkit-progress-bar { background: var(--border); border-radius: 4px; }
    progress::-webkit-progress-value { background: var(--accent); border-radius: 4px; transition: width .25s ease; }
    progress::-moz-progress-bar { background: var(--accent); border-radius: 4px; }
    #progress-label { display: block; margin-top: 6px; font-size: 13px; opacity: .68; text-align: center; }
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
const card = document.getElementById("card");
const zone = document.getElementById("dropzone");
const input = document.getElementById("file");
const status = document.getElementById("status");
const bar = document.getElementById("bar");
const progressLabel = document.getElementById("progress-label");

function setStatus(msg, kind = "") { status.textContent = msg; status.className = kind; }
function setProgress(value, page, total) { bar.value = value; progressLabel.textContent = `Page ${page} of ${total}`; }
function resetProgress() { bar.value = 0; progressLabel.textContent = ""; }

["dragenter", "dragover"].forEach(n => zone.addEventListener(n, e => { e.preventDefault(); zone.classList.add("dragging"); }));
["dragleave", "drop"].forEach(n => zone.addEventListener(n, e => { e.preventDefault(); zone.classList.remove("dragging"); }));
zone.addEventListener("drop", e => { const f = e.dataTransfer.files[0]; if (f) upload(f); });
input.addEventListener("change", () => { const f = input.files[0]; if (f) upload(f); });

async function upload(file) {
  if (!file.name.toLowerCase().endsWith(".pdf")) { setStatus("Choose a PDF file.", "error"); return; }
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
          const a = document.createElement("a");
          a.href = `/download/${event.token}`;
          a.download = event.filename;
          document.body.appendChild(a); a.click(); a.remove();
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


@app.get("/download/{token}")
def download(token: str) -> Response:
    entry = _DOWNLOAD_STORE.pop(token, None)
    if entry is None:
        raise HTTPException(404, "Download link not found or already used.")
    pdf_bytes, filename, expires_at = entry
    if time.monotonic() > expires_at:
        raise HTTPException(410, "Download link has expired.")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/health")
def health() -> dict[str, str]:
    from detection import lid_model, pii_model
    try:
        labels, _ = lid_model.predict("health check", k=1)
        if not labels:
            raise RuntimeError("lid_model returned no predictions")
        if pii_model is None:
            raise RuntimeError("pii_model not loaded")
    except Exception as exc:
        raise HTTPException(503, f"Models not ready: {exc}") from exc
    return {"status": "ok"}


def clean_filename(name: str) -> str:
    base = Path(name or "ticket.pdf").stem
    return (re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._") or "ticket")[:100]


async def read_limited(upload: UploadFile) -> bytes:
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"PDF exceeds the {MAX_UPLOAD_BYTES // 1024 // 1024} MB limit.")
    return data


@app.post("/anonymize")
async def anonymize(file: UploadFile = File(...)) -> StreamingResponse:
    filename = file.filename or "ticket.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(415, "Only PDF files are accepted.")

    pdf_bytes = await read_limited(file)
    if not pdf_bytes.startswith(b"%PDF-"):
        raise HTTPException(415, "The uploaded file is not a valid PDF.")

    fingerprint = hashlib.sha256(pdf_bytes).hexdigest()[:12]
    logger.info("Processing PDF fingerprint=%s bytes=%d", fingerprint, len(pdf_bytes))

    pages, page_sizes = await asyncio.to_thread(extract_pages, pdf_bytes)
    output_name = f"{clean_filename(filename)}_anonymized.pdf"

    async def generate():
        total = len(pages)
        pseudonymizer = Pseudonymizer()
        output_pages: list[str] = []
        page_metadata: list[dict[str, object]] = []

        try:
            async with _NER_SEM:
                for i, text in enumerate(pages):
                    language, confidence = await asyncio.to_thread(detect_language, text)
                    spans = normalize_spans(text,
                        await asyncio.to_thread(regex_spans, text) +
                        await asyncio.to_thread(model_spans, text)
                    )
                    output_pages.append(pseudonymizer.apply(text, spans))
                    page_metadata.append({
                        "page": i + 1,
                        "language": language,
                        "language_confidence": round(confidence, 3),
                        "entities": len(spans),
                    })
                    yield f'data: {{"progress":{round((i + 1) / total, 3)},"page":{i + 1},"total":{total}}}\n\n'

                output_pages = await asyncio.to_thread(consistency_sweep, output_pages, pseudonymizer)
                result = await asyncio.to_thread(build_pdf, output_pages, page_metadata, page_sizes[0])

            logger.info(
                "Finished PDF fingerprint=%s pages=%d replacements=%d",
                fingerprint, total,
                sum(int(m["entities"]) for m in page_metadata),
            )
            token = _new_download_token(result, output_name)
            yield f'data: {{"done":true,"filename":{json.dumps(output_name)},"token":{json.dumps(token)}}}\n\n'
        except Exception as exc:
            logger.exception("Error processing PDF fingerprint=%s", fingerprint)
            yield f'data: {json.dumps({"error": str(exc) or "An unexpected error occurred."})}\n\n'

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store, max-age=0", "X-Accel-Buffering": "no"},
    )
