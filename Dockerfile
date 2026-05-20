FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

WORKDIR /app

# Install dependencies first — cached until lockfile changes
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --all-extras

COPY src/ ./src/
COPY config.yaml ./
COPY data/ ./data/

RUN mkdir -p logs

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "src/vortosql/ui.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
