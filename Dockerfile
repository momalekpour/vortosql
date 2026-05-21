FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

WORKDIR /app

RUN groupadd --system --gid 1000 appuser \
 && useradd  --system --uid 1000 --gid appuser --home-dir /app --no-create-home appuser

# Install dependencies first — cached until lockfile changes
COPY --chown=appuser:appuser pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev

COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser config.yaml ./
COPY --chown=appuser:appuser data/ ./data/

RUN mkdir -p logs && chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request, sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3).status == 200 else 1)" \
    || exit 1

CMD ["uv", "run", "streamlit", "run", "src/vortosql/ui.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
