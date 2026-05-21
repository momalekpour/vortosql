# NL2SQL Data Agent

## Setup

```bash
# Clone the repository
git clone https://github.com/momalekpour/vortosql.git
cd vortosql

# Install dependencies (uv manages the virtualenv and Python version automatically)
# Install uv first if needed: https://docs.astral.sh/uv/getting-started/installation/
uv sync

# Optional extras — only needed if you use the HuggingFace provider or few-shot examples:
#   uv sync --extra huggingface   # transformers (~1 GB with torch)
#   uv sync --extra examples      # datasets (BIRD few-shot loader)

# (Optional) Install pre-commit hooks for auto linting (ruff) and formatting (black) for development
uv run pre-commit install
```

## Run

### Local

```bash
# Copy the example env file and set your OPENAI_API_KEY
cp .env.example .env

# Web UI (recommended) 
bash scripts/run_ui.sh

# CLI REPL
bash scripts/run_cli.sh
```

### Docker

```bash
cp .env.example .env   # fill in OPENAI_API_KEY

# Web UI → http://localhost:8501
docker compose up ui

# Interactive CLI
docker compose run --rm cli

# PostgreSQL only (for development/testing)
docker compose up postgres -d
```

## Architecture

### Pipeline Overview

See [`docs/architecture.md`](docs/architecture.md) for full details. The application is built around a **composable operator pipeline** configured via `config.yaml` — each operator's model provider, technique, and behaviour is plug-and-play; the default setup is ready to run as-is. Each operator implements an `execute(context)` method that reads from and writes to a shared context dictionary. The pipeline runs the following operators in order:

1. **IntentGuardrail** - Optional LLM-based scope classifier; rejects out-of-scope questions via early-stop. Skipped when no scope is configured.
2. **SchemaLinker** - Resolves which tables/columns are relevant to the question
3. **ExampleSelector** - Retrieves similar few-shot examples (skipped in zero-shot mode)
4. **SQLGenerator** - LLM generates a SQL query from the question, schema, and examples
5. **SQLCorrector** - Validates and auto-corrects SQL errors via retry loop
6. **SQLExecutor** - Executes the final SQL against the database
7. **AnswerGenerator** - LLM summarises the query results into a natural language answer

### Guardrails

Two opt-in defences, both controlled by `config.yaml` (or by overrides passed to `NL2SQLApp`):

- **Layer 0 — Intent gate** (IntentGuardrail, soft): when `scope` is set, the LLM checks whether the question fits the described scope and triggers an early-stop if not. `scope: null` disables the operator entirely.
- **Layer 1 — Schema restriction** (SchemaLinker, hard): when `schema_guardrails` is set, only allowlisted tables/columns are exposed to the LLM. `schema_guardrails: null` exposes the full schema.

### Session Logs

Opt-in: set `VORTOSQL_DUMP_SESSION_LOGS=1` and every pipeline run will write its full config and context to `logs/<timestamp>.json` for traceability.
