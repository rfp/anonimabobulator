from __future__ import annotations

import logging
import os
import re
import warnings
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("anonimabobulator")
logging.getLogger("gliner").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="gliner")

MAX_UPLOAD_BYTES = int(os.getenv("APP_MAX_UPLOAD_MB", "30")) * 1024 * 1024
LID_MODEL_PATH   = os.getenv("APP_LID_MODEL", "/opt/models/lid.176.ftz")
MIN_TEXT_CHARS   = 80
MODEL_THRESHOLD  = float(os.getenv("APP_MODEL_THRESHOLD", "0.45"))
# GLiNER's DeBERTa encoder caps at 384 tokens. German text ≈ 3 chars/token,
# so 1800 chars was producing 500-600 token chunks and silently truncating.
# 800 chars stays comfortably under the limit across all supported languages.
CHUNK_SIZE       = int(os.getenv("APP_CHUNK_SIZE", "800"))
CHUNK_OVERLAP    = int(os.getenv("APP_CHUNK_OVERLAP", "80"))
OCR_ENABLED      = os.getenv("APP_OCR_ENABLED", "false").lower() in ("1", "true", "yes")


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
                        # Casefold the pattern so matching against .casefold() values
                        # works correctly for Unicode characters like ß → ss.
                        patterns.append(re.compile(line[1:].casefold(), re.IGNORECASE | re.UNICODE))
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
