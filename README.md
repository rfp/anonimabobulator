# Anonimabobulator

A self-contained, offline PDF anonymization service. Upload a PDF through a browser, receive an anonymized copy. Everything runs inside a single Docker container — no data leaves the machine.

## Features

- **Fully offline** — models are baked into the image at build time; no network calls at runtime
- **Multilingual** — fastText detects the document language; GLiNER (`urchade/gliner_multi_pii-v1`) handles PII in 50+ languages
- **Consistent pseudonyms** — the same name always gets the same token (`[PERSON_001]`, `[EMAIL_001]`, …) across all pages
- **Two detection layers** — a high-confidence regex layer (emails, IPs, IBANs, contract IDs, …) and a neural NER layer
- **Consistency sweep** — a second pass catches occurrences the model missed on earlier pages
- **Whitelist support** — define terms or regex patterns that should never be anonymized
- **Privacy by design** — originals never written to disk; logs contain only a SHA-256 fingerprint and byte size, never content
- **Non-root container** — runs as uid 10001

## Quick start

```bash
# Build (downloads models — takes a few minutes; subsequent builds are fast)
docker build -t anonimabobulator .

# Run
docker run --rm -p 127.0.0.1:8000:8000 --name anonimabobulator anonimabobulator
```

Open **http://localhost:8000**, drop a PDF, and the anonymized version downloads automatically.

## What gets anonymized

| Category | Examples |
|---|---|
| Person names | full names, standalone surnames after honorifics (Herr/Frau) |
| Contact | email addresses, phone numbers |
| Network | IP addresses (v4/v6), MAC addresses, hostnames, URLs |
| Financial | IBANs, credit card numbers (Luhn-validated), bank accounts |
| Identity | passport, national ID, driver's licence, tax ID, health insurance number |
| Credentials | API keys, tokens, passwords, private keys |
| References | contract numbers, customer IDs, serial numbers, licence keys |
| Other | dates of birth, postal codes, social media handles |

## Whitelist

Terms and patterns listed in `whitelist-*.txt` files are never anonymized.

```bash
cp whitelist-example.txt whitelist-myproject.txt
# edit whitelist-myproject.txt, then rebuild
docker build -t anonimabobulator .
```

**File format** — one entry per line; lines starting with `#` are comments:

```
# exact match (case-insensitive)
Acme Corp
Support Team

# regex full-match (line starts with ~)
~\d{4}\.\d+\.\d+
```

Multiple `whitelist-*.txt` files are all loaded. The `whitelist-example.txt` file is a documented template — copy and rename it.

> **Note:** whitelist files are baked into the image at build time. Rebuild after any change.

## Configuration

Environment variables (all optional):

| Variable | Default | Description |
|---|---|---|
| `APP_MAX_UPLOAD_MB` | `30` | Maximum PDF upload size in MB |
| `APP_MODEL_THRESHOLD` | `0.45` | GLiNER confidence threshold (lower = more aggressive) |
| `APP_CHUNK_SIZE` | `800` | Characters per NER chunk (keep ≤ ~1000 to avoid model truncation) |
| `APP_CHUNK_OVERLAP` | `80` | Overlap between chunks |

```bash
docker run --rm -p 127.0.0.1:8000:8000 \
  -e APP_MODEL_THRESHOLD=0.35 \
  anonimabobulator
```

## Technology

| Component | Library |
|---|---|
| Web service | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn |
| NER / PII detection | [GLiNER](https://github.com/urchade/GLiNER) (`urchade/gliner_multi_pii-v1`) |
| Language detection | [fastText](https://fasttext.cc/) (`lid.176.ftz`, 176 languages) |
| PDF text extraction | [PyMuPDF](https://pymupdf.readthedocs.io/) |
| PDF generation | [ReportLab](https://www.reportlab.com/) |

The entire application lives in a single `Dockerfile` — no build context files beyond optional whitelists.

## Limitations

- **Text-layer PDFs only** — scanned documents without an embedded text layer are not supported. Export to PDF using "Print to PDF" or a PDF generator, not a scanner.
- **Layout is not preserved** — the output is a plain-text reformatted PDF, not a pixel-perfect copy of the original.

## License

MIT — see [LICENSE](LICENSE).
