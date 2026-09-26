# EMVA internal tool: Streamlit app (app/main.py) + the emva package, for Railway or `make docker`.
FROM python:3.11-slim

ENV DATA_DIR=/data \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN useradd --uid 1000 --create-home --home-dir /home/app --shell /usr/sbin/nologin app \
    && mkdir -p /data \
    && chown app:app /data

WORKDIR /app

# Dependencies first so source edits reuse this layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY Makefile ./
COPY emva/ emva/
COPY baseline/ baseline/
COPY scripts/ scripts/
COPY app/ app/
# Built-in dataset mappings (Map & convert's "Use saved mapping" picker, app.ingest.BUILTIN_MAPPINGS_DIR).
COPY mappings/ mappings/
# Streamlit theme (Keel palette and fonts); read from the working directory.
COPY .streamlit/ .streamlit/
# Sample dataset for the first-run demo; ground_truth* is excluded by .dockerignore.
COPY data/v1/ data/v1/

USER app

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8080\")}/_stcore/health', timeout=4)"

CMD ["sh", "/app/scripts/serve.sh"]
