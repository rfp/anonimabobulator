# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/models/huggingface \
    TOKENIZERS_PARALLELISM=false \
    APP_MAX_UPLOAD_MB=30 \
    APP_LID_MODEL=/opt/models/lid.176.ftz

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl \
      fonts-dejavu-core \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
      "fastapi>=0.115,<1" \
      "uvicorn[standard]>=0.34,<1" \
      "python-multipart>=0.0.20,<1" \
      "pymupdf>=1.25,<2" \
      "reportlab>=4.2,<5" \
      "fasttext-wheel>=0.9.2,<1" \
      "gliner>=0.2.16,<1" \
      "huggingface-hub>=0.30,<1"

# Download both models during image build so runtime processing is fully offline.
# GLiNER.from_pretrained (not snapshot_download) ensures the tokenizer and encoder
# backbone are also cached transitively.
RUN mkdir -p /opt/models \
    && curl -fsSL \
       https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz \
       -o /opt/models/lid.176.ftz \
    && python -c "from gliner import GLiNER; GLiNER.from_pretrained('urchade/gliner_multi_pii-v1')"

WORKDIR /app

COPY *.py whitelist-*.txt ./

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app /opt/models

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
