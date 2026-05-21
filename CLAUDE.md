# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync --all-extras

# Run CLI (REPL)
bash scripts/run_cli.sh

# Run web UI (Streamlit)
bash scripts/run_ui.sh

# Lint
uv run ruff check src/

# Format
uv run black src/

# Run tests (DB adapters + pipeline operators with mocked LLMs)
uv run pytest tests/

# Ad-hoc full-pipeline smoke (calls real LLM; requires OPENAI_API_KEY)
uv run python scripts/smoke_pipeline.py
```

Tests live in `tests/`: DB adapter unit tests in `tests/core/database/` and pipeline operator tests with mocked LLMs in `tests/pipeline/`. Pre-commit hooks run ruff + black automatically.

## Docker

```bash
# Build image
docker compose build

# Run web UI  →  http://localhost:8501
docker compose up ui

# Run interactive CLI
docker compose run --rm cli

# Start PostgreSQL only (for adapter dev/tests)
docker compose up postgres -d

# Run all services
docker compose up
```

Secrets (`OPENAI_API_KEY`, etc.) are read from `.env` at runtime — they are never baked into the image. Copy `.env.example` → `.env` and fill in your key before running.

## Architecture

This is a **modular NL2SQL pipeline** that translates natural language questions into SQL, optionally enforces an intent gate and a schema allowlist, executes SQL, and returns a natural language answer.

### Entry Points

- `cli.py` — Interactive terminal REPL; loops on user questions
- `ui.py` — Streamlit web UI with chat history
- `app.py:NL2SQLApp` — Session manager; loads `config.yaml` and delegates to `NL2SQLPipeline`. Accepts optional `scope` and `schema_guardrails` overrides

### Pipeline Operator Chain

All pipeline steps inherit from `Operator` (ABC in `pipeline/operator.py`). They share a **mutable context dict** — each operator reads its inputs from the context and writes outputs back. Pipeline halts early if any operator sets `context["pipeline_early_stop"] = True`.

```
NL2SQLApp.ask(question)
  → NL2SQLPipeline.execute(question, scope, schema_guardrails)
      1. IntentGuardrail   — LLM scope check; skipped if scope is null
      2. SchemaLinker      — Extracts DB schema; filters via schema_guardrails (if set)
      3. ExampleSelector   — Few-shot retrieval from BIRD dataset (skipped for zero_shot)
      4. SQLGenerator      — Jinja2 prompt → LLM generates SQL
      5. SQLCorrector      — sqlglot validation + LLM retry (skipped if max_correction_attempts=0)
      6. SQLExecutor       — Executes SQL against the database
      7. AnswerGenerator   — LLM summarizes results into natural language
```

### Guardrails

Two opt-in defences, both sourced from `config.yaml` (overridable via `NL2SQLApp` constructor args):

| Layer | Where | Type | Mechanism |
|-------|-------|------|-----------|
| 0 | IntentGuardrail | Soft | When `scope` is set, the LLM classifies whether the question is in-scope. `scope: null` skips the operator entirely. |
| 1 | SchemaLinker | Hard | When `schema_guardrails` is set, only allowlisted tables/columns are shown to the LLM. `schema_guardrails: null` exposes the full schema. Use `"*"` to allow all columns of a table. |

### Key Abstractions

- **`Operator` ABC** — `__init__(config: dict)` + `execute(context: dict) -> None`; config is static, context is mutable and shared
- **`NL2SQLPipelineConfig`** (Pydantic, `pipeline/config.py`) — aggregates per-operator config classes; validated at startup from `config.yaml`
- **`ModelManager`** — factory for LLM providers (OpenAI, Anthropic, Ollama, HuggingFace); operators call `ModelManager.create_model(config)`
- **`DatabaseHandler`** — adapter pattern over SQLite/DuckDB; swap via config
- **`PromptRenderer`** — Jinja2 with `StrictUndefined`; templates live per-operator at `src/vortosql/pipeline/<operator>/prompt_templates/*.jinja`
- **`Logger`** — structured JSON via loguru: `logger.log("info", "EVENT_NAME", {...})`

### Configuration

`config.yaml` at the repo root controls everything. Three cross-cutting keys live directly under `nl2sql_pipeline`:

- `db_file_path` — shared DB path, injected into SchemaLinker and SQLExecutor at build time
- `intent_guardrail.scope` — optional scope description for the intent gate; `null` disables it
- `schema_linker.schema_guardrails` — optional table/column allowlist; `null` exposes the full schema

Everything else (model provider, temperature, prompt template, `max_correction_attempts`) is grouped per operator. Changes to `config.yaml` take effect without code changes.

### Source Layout

```
src/vortosql/
├── pipeline/          # Operator chain: each operator dir holds its code + prompt_templates/
│   ├── operator.py    # Operator ABC
│   ├── config.py      # Pydantic config classes for all operators
│   ├── nl2sql_pipeline.py  # Orchestrator
│   ├── intent_guardrail/   # code + prompt_templates/intent_check.jinja
│   ├── schema_linker/      # code + prompt_templates/{SCSL,TCSL_*}.jinja
│   ├── example_selector/
│   ├── sql_generator/      # code + prompt_templates/{zero_shot,few_shot}.jinja
│   ├── sql_corrector/      # code + prompt_templates/syntax_correction.jinja
│   ├── sql_executor/
│   └── answer_generator/   # code + prompt_templates/answer.jinja
├── core/
│   ├── database/      # SQLite + DuckDB adapters
│   ├── model_manager/ # LLM provider factory
│   ├── logger/        # Structured JSON logger
│   └── prompt_renderer/  # Jinja2 renderer
├── app.py             # NL2SQLApp session manager
├── cli.py             # Terminal REPL
└── ui.py              # Streamlit UI
```

Full technical documentation: `docs/architecture.md`
