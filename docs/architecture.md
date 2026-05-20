# Architecture

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Request Flow](#request-flow)
- [Operator Reference](#operator-reference)
- [Guardrails](#guardrails)
- [Application Layer](#application-layer)
- [Core Utilities](#core-utilities)
- [Docker](#docker)
- [Configuration Reference](#configuration-reference)

---

## Overview

A modular NL2SQL system that translates natural language questions into SQL queries, optionally gates them with an intent classifier and a schema allowlist, executes the SQL against a SQLite database, and returns results in natural language. Built around a composable operator pipeline with two opt-in guardrail layers and a generic early-stop mechanism.

---

## Project Structure

```
src/vortosql/
├── __main__.py                     # Module entry point — allows `python -m vortosql`
├── app.py                          # Session orchestration — builds pipeline, exposes ask()
├── cli.py                          # Interactive terminal REPL with formatted table output
├── ui.py                           # Streamlit web UI (landing page + chat)
│
├── core/
│   ├── database/                   # DB connection and query execution
│   │   ├── database_handler.py     # Unified interface; factory over adapters
│   │   └── adapters/
│   │       ├── base_adapter.py     # ABC: connect(), run_query(), close_connection()
│   │       ├── sqlite_adapter.py
│   │       ├── duckdb_adapter.py
│   │       └── postgres_adapter.py
│   ├── logger/                     # Structured JSON logger (loguru)
│   │   └── logger.py               # Logger(name).log(level, event, payload)
│   ├── model_manager/              # LLM provider abstraction
│   │   ├── model_manager.py        # Factory: ModelManager.create_model(provider, type, name)
│   │   ├── openai_model.py         # Chat completion + embeddings (OpenAI API)
│   │   ├── ollama_model.py         # Local models via Ollama
│   │   ├── anthropic_model.py      # Anthropic Claude (chat only)
│   │   ├── huggingface_model.py    # HuggingFace inference
│   │   └── utils.py                # compose_chat_messages helper
│   └── prompt_renderer/            # Jinja2 template renderer (StrictUndefined)
│       └── prompt_renderer.py      # PromptRenderer(templates_dir_path).render(name, context)
│
└── pipeline/
    ├── operator.py                 # Operator ABC: __init__(config), execute(context)
    ├── config.py                   # Pydantic config models for every operator + pipeline
    ├── nl2sql_pipeline.py          # Assembles and runs the operator chain; early-stop loop
    ├── intent_guardrail/          # Step 1: optional LLM scope gate — rejects out-of-scope questions
    ├── schema_linker/              # Step 2: schema extraction and optional LLM filtering
    ├── example_selector/           # Step 3 (optional): few-shot examples from BIRD dataset
    ├── sql_generator/              # Step 4: NL → SQL via LLM + Jinja2 prompt
    ├── sql_corrector/              # Step 5 (optional): syntax correction loop with sqlglot
    ├── sql_executor/               # Step 6: DB execution
    └── answer_generator/           # Step 7: LLM summarises results into natural language

config.yaml                         # Runtime configuration for all operators
Dockerfile                          # Single-stage image (uv + Python 3.14); defaults to Streamlit UI
docker-compose.yml                  # Services: ui (8501), cli (interactive), postgres (5432)
.dockerignore                       # Excludes .env, caches, logs from build context

scripts/
├── run_ui.sh                       # Sources .env, then launches Streamlit
├── run_cli.sh                      # Sources .env, then launches the CLI
└── load_dotenv.sh                  # Exports all vars from .env into the shell

data/
└── employees.db                    # SQLite database (demo data)

docs/
└── architecture.md                 # This file
```

---

## Request Flow

```
User input (cli.py / ui.py)
    │
    ▼
NL2SQLApp.ask(user_question)          ← app.py
    │  forwards constructor overrides (scope, schema_guardrails) if set
    ▼
NL2SQLPipeline.execute(user_question, scope, schema_guardrails)
    │  effective scope / schema_guardrails are the override args
    │  if provided, else the values from config.yaml
    │
    │  context = {user_question, scope, schema_guardrails}  ← initial context
    │
    ├─► IntentGuardrail.execute(context)
    │       Skips entirely if `scope` is falsy.
    │       Otherwise LLM scope check against the configured `scope`.
    │       writes: intent_guardrail_is_in_scope, intent_guardrail_reason
    │       if out of scope → sets pipeline_early_stop → pipeline breaks here
    │
    ├─► SchemaLinker.execute(context)
    │       writes: schema_linker_db_schema, schema_linker_db_columns
    │
    ├─► ExampleSelector.execute(context)   [only when prompt_template = few_shot]
    │       writes: example_selector_examples
    │
    ├─► SQLGenerator.execute(context)
    │       writes: sql_query, sql_generator_sql_query, sql_generator_prompt, LLM metadata
    │
    ├─► SQLCorrector.execute(context)      [only when max_correction_attempts > 0]
    │       writes: sql_query (corrected), sql_corrector_sql_query, sql_corrector_is_successful
    │
    ├─► SQLExecutor.execute(context)
    │       reads: sql_query
    │       executes SQL against DB
    │       writes: sql_executor_sql_query, sql_executor_columns, sql_executor_rows,
    │               sql_executor_row_count, sql_executor_error
    │
    └─► AnswerGenerator.execute(context)
            reads: user_question, sql_executor_columns, sql_executor_rows, sql_executor_sql_query
            skips if: early_stop, sql_executor_error, or row_count == 0
            LLM summarises the query results into a natural language answer
            writes: answer_generator_answer, answer_generator_prompt, LLM metadata
    │
    ▼
context dict returned (+ pipeline_latency, timestamp)
```

All operators share a single mutable `context` dict. Each operator reads its inputs from context and writes its outputs back to it. Operator config is static (set at construction); per-request data flows through context only.

**Early-stop mechanism:** After each operator, the pipeline checks `context.get("pipeline_early_stop")`. If truthy, the loop breaks and the remaining operators are skipped. The string value is the human-readable reason. Any operator can trigger this — it is not specific to `IntentGuardrail`.

---

## Operator Reference

### 1. IntentGuardrail

The first operator in the pipeline. Uses an LLM to determine whether the user's question is within the configured scope before any expensive downstream work (schema reading, SQL generation) is done.

**Config (`IntentGuardrailConfig`):**

| Key | Type | Required |
|---|---|---|
| `chat_completion_model_provider` | `ModelProvider` | yes |
| `chat_completion_model_name` | model enum | yes |
| `temperature` | `float [0, 2]` | yes |

The scope description is an operator-level config key (`nl2sql_pipeline.intent_guardrail.scope` in `config.yaml`), overridable at construction time via `NL2SQLApp(scope=...)`. When `scope` is empty/null, IntentGuardrail short-circuits: it writes `intent_guardrail_is_in_scope=True` with reason `"no scope configured"` and returns without calling the LLM.

**Prompt template (`intent_check.jinja`):**

A generic, scope-parameterised classifier. Receives `scope` and `user_question`, asks the LLM to return `{"is_in_scope": bool, "reason": str}`.

**Behaviour:**
- Parses the JSON response; strips markdown fences if present
- On JSON parse failure: **fails open** (`is_in_scope = True`) — false negatives are preferable to blocking valid questions
- When out of scope: sets `context["pipeline_early_stop"]` with a user-facing message, halting the pipeline immediately

**Context writes:** `intent_guardrail_is_in_scope` (bool), `intent_guardrail_reason` (str), `pipeline_early_stop` (str, only when out of scope)

---

### 2. SchemaLinker  *(schema restriction)*

Connects to the SQLite database at startup, reads the full schema (tables, columns, PKs, FKs) once via PRAGMA queries, and produces a textual schema description for downstream operators.

**Config (`SchemaLinkerConfig`):**

| Key | Type | Required |
|---|---|---|
| `db_file_path` | `str` | yes |
| `technique` | `SchemaLinkingTechnique` | yes |
| `model_provider` | `ModelProvider` | for TCSL/SCSL only |
| `model_name` | `OpenAIModel \| OllamaModel` | for TCSL/SCSL only |

Pydantic `model_validator` enforces that `model_provider` and `model_name` are present when technique is TCSL or SCSL.

**Techniques:**

| Technique | Description | LLM calls |
|---|---|---|
| `full` | Returns all tables and columns verbatim | 0 |
| `tcsl` | Two-step: LLM picks relevant tables, then relevant columns | 2 |
| `scsl` | Scores each column individually | N (one per column) |

**Guardrail — Layer 1 (schema restriction):**

Reads `schema_guardrails` from its own operator config (`nl2sql_pipeline.schema_linker.schema_guardrails` in `config.yaml`), overridable via `NL2SQLApp(schema_guardrails=...)`. When set, `_apply_schema_guardrails()` filters `self.tables` to only the allowed tables/columns using shallow copies — `self.tables` is never mutated. Pass `"*"` to allow all columns in a table. Tables absent from `schema_guardrails` are invisible to the LLM entirely. When `schema_guardrails` is `None`, the full schema is exposed.

**Context writes:** `schema_linker_db_schema` (str), `schema_linker_db_columns` (dict[str, list[str]])

---

### 3. ExampleSelector *(optional)*

Selects few-shot SQL examples from the BIRD mini-dev HuggingFace dataset to include in the SQL generation prompt. On first run, the dataset is automatically downloaded from HuggingFace — this is a high-quality collection of NL→SQL pairs used as ground-truth examples for few-shot prompting. Subsequent runs use the local cache at `data/cache/huggingface/datasets/`.

Skipped entirely when `sql_generator.prompt_template = zero_shot` (the operator is not added to the operator list).

**Config (`ExampleSelectorConfig`):**

| Key | Type | Required |
|---|---|---|
| `technique` | `ExampleSelectionTechnique` | yes |
| `number_of_examples` | `int` | yes |
| `embedding_model_provider` | `ModelProvider` | for `question_similarity` |
| `embedding_model_name` | model enum | for `question_similarity` |
| `random_seed` | `int \| None` | no |

Pydantic `model_validator` enforces embedding fields when technique is `question_similarity`.

**Techniques:**

| Technique | Description |
|---|---|
| `random` | Randomly samples N examples; respects `random_seed` for reproducibility |
| `question_similarity` | Embeds the user question, returns N nearest-neighbour examples by cosine similarity |

**Context writes:** `example_selector_examples`

---

### 4. SQLGenerator

Renders a Jinja2 prompt template with the schema, examples (if any), and user question, then calls an LLM for SQL generation.

**Config (`SQLGeneratorConfig`):**

| Key | Type | Required |
|---|---|---|
| `prompt_template` | `SQLGenerationPromptTemplate` | yes |
| `chat_completion_model_provider` | `ModelProvider` | yes |
| `chat_completion_model_name` | model enum | yes |
| `temperature` | `float [0, 2]` | yes |
| `random_seed` | `int \| None` | no |

**Templates:**

| Template | Context variables used |
|---|---|
| `zero_shot` | `schema_linker_db_schema`, `user_question` |
| `few_shot` | above + `example_selector_examples` |

Post-processing: LLM output is stripped of markdown fences (` ```sql ``` `) and collapsed to a single line.

**Context writes:** `sql_query`, `sql_generator_sql_query`, `sql_generator_prompt`, plus prefixed LLM metadata keys (`sql_generator_model`, `sql_generator_latency`, `sql_generator_prompt_tokens`, etc.)

---

### 5. SQLCorrector *(optional)*

Validates the generated SQL by parsing it with sqlglot. If parsing fails, sends the SQL + error back to an LLM for correction. Repeats up to `max_correction_attempts` times.

Skipped when `max_correction_attempts = 0` (operator not added to operator list).

**Config (`SQLCorrectorConfig`):**

| Key | Type | Required |
|---|---|---|
| `prompt_template` | `SQLCorrectionPromptTemplate` | yes |
| `max_correction_attempts` | `int` | yes |
| `dbms` | `DBMS` | yes |
| `chat_completion_model_provider` | `ModelProvider` | yes |
| `chat_completion_model_name` | model enum | yes |
| `temperature` | `float [0, 2]` | yes |
| `random_seed` | `int \| None` | no |

**Context writes:** `sql_query` (corrected in-place), `sql_corrector_sql_query`, `sql_corrector_is_successful`, `sql_corrector_prompt`, `sql_corrector_num_attempts`, `sql_corrector_latency`, `sql_corrector_num_input_tokens`, `sql_corrector_num_output_tokens`

---

### 6. SQLExecutor

Executes the generated SQL against the configured database and writes the results back to context.

**Config (`SQLExecutorConfig`):**

| Key | Type | Required |
|---|---|---|
| `db_file_path` | `str` | yes |
| `dbms` | `DBMS` | yes |

On execution error: writes the error string to `sql_executor_error` and zero-fills the result fields; does not raise — callers handle gracefully.

**Context writes:** `sql_executor_sql_query`, `sql_executor_columns`, `sql_executor_rows`, `sql_executor_row_count`, `sql_executor_error`

---

### 7. AnswerGenerator

The final operator. Takes the user's original question and the SQL query results, sends them to an LLM, and produces a natural language answer.

Skipped when the pipeline was early-stopped (out-of-scope question), when SQL execution failed, or when the query returned zero rows — in those cases no LLM call is made and no answer is written.

**Config (`AnswerGeneratorConfig`):**

| Key | Type | Required |
|---|---|---|
| `chat_completion_model_provider` | `ModelProvider` | yes |
| `chat_completion_model_name` | model enum | yes |
| `temperature` | `float [0, 2]` | yes |

**Prompt template (`answer.jinja`):**

Receives `user_question`, `sql_executor_sql_query`, `sql_executor_columns`, and `sql_executor_rows`. Formats the results as a readable list and asks the LLM to answer the question based on the data. Instructs the LLM to keep the answer brief and avoid SQL or technical details.

**Context writes:** `answer_generator_answer` (str), `answer_generator_prompt` (str), plus prefixed LLM metadata keys

---

## Guardrails

Two opt-in defences, both controlled at the pipeline level (via `config.yaml` or `NL2SQLApp` constructor overrides):

| Layer | Operator | Mechanism | Strength |
|---|---|---|---|
| 0 — Intent gate | IntentGuardrail | When `scope` is set, the LLM classifies whether the question fits the described scope. Out-of-scope questions halt the pipeline before any SQL is generated. `scope: null` skips the operator entirely. | Soft — LLM-dependent; fails open on parse error to avoid blocking valid questions |
| 1 — Schema restriction | SchemaLinker | `schema_guardrails` hides entire tables/columns from the LLM. It never sees what isn't in the allowlist. `schema_guardrails: null` exposes the full schema. | Hard — the LLM cannot reference what it cannot see |

**Examples:**

- **Layer 0 (intent gate):** `scope: "Questions about HR records."` is set. User asks *"What's the weather?"* → LLM flags out of scope → pipeline stops immediately, no SQL generated.

- **Layer 1 (schema restriction):** `schema_guardrails = {"Employee": ["*"]}` → every other table is hidden. The LLM physically cannot reference them — they don't exist in the schema it sees.

When `scope: null`, Layer 0 is bypassed entirely (no LLM call is made by IntentGuardrail). When `schema_guardrails: null`, the LLM is shown the full database schema.

---

## Application Layer

### `app.py` — `NL2SQLApp`

Owns one session:

- `__init__(config_path="config.yaml", scope=None, schema_guardrails=None)`: Loads `config.yaml` and constructs `NL2SQLPipeline`. When `scope` or `schema_guardrails` are provided, they overwrite the corresponding values in the loaded config (`intent_guardrail.scope`, `schema_linker.schema_guardrails`) before the pipeline is built.
- `ask(user_question) -> dict`: Delegates to `pipeline.execute(user_question, scope=..., schema_guardrails=...)` and returns the full context dict.

### `cli.py` — Terminal REPL

- Instantiates `NL2SQLApp()` (no per-session arguments)
- Input loop: prompts for question → calls `app.ask()` → checks `pipeline_early_stop` first (prints the message and continues) → otherwise renders ASCII table via `_format_table()`
- Handles `exit`/`quit`, empty input, EOF, and KeyboardInterrupt cleanly

### `ui.py` — Streamlit UI

Two-phase flow:

1. **Landing page** (`"started" not in session_state`): Hero section with title, subtitle, and "Get Started →" button. On click: sets `started=True`, empty history, reruns.
2. **Chat page**: Replays `session_state["history"]` top-to-bottom (oldest first) as user/assistant message pairs. New questions submitted via `st.chat_input`. Each history entry is rendered as:
   - `st.warning(...)` if `early_stop` is set (out-of-scope rejection)
   - `st.error(...)` if a SQL execution error occurred
   - `st.info("No results found.")` for empty result sets
   - Natural language answer (from `answer_generator_answer`) + `st.dataframe(...)` with SQL + row count + latency caption otherwise

`@st.cache_resource` ensures the `NL2SQLApp` instance is reused across reruns.

---

## Core Utilities

### DatabaseHandler

Adapter pattern over SQLite and DuckDB:

```python
# SQLite / DuckDB
db = DatabaseHandler(DBMS.SQLITE, {"db_path": "data/employees.db"})

# PostgreSQL
db = DatabaseHandler(DBMS.POSTGRES, {
    "host": "localhost", "port": 5432,
    "dbname": "vortosql", "user": "vortosql", "password": "vortosql",
})

columns, rows = db.run_query("SELECT ...")
db.close_connection()
```

`DatabaseHandler.__init__` calls `connect()` immediately. Always call `close_connection()` in a `finally` block.

### ModelManager

Factory for LLM clients:

```python
llm = ModelManager.create_model(
    model_provider=ModelProvider.OPENAI,
    model_type=ModelType.COMPLETION,
    model_name=OpenAIModel.GPT_54_MINI,
)
response = llm.get_chat_completion(messages=..., temperature=0)
```

`ModelType.EMBEDDING` is also supported for providers that offer it.

### PromptRenderer

Jinja2 renderer with `StrictUndefined` (missing variables raise at render time, not silently):

```python
renderer = PromptRenderer(templates_dir_path="path/to/templates")
prompt = renderer.render("zero_shot", context)  # renders zero_shot.jinja
```

### Logger

Structured JSON logging via loguru:

```python
logger = Logger(__name__)
logger.log("info", "EVENT_NAME", {"key": "value"})
```
---

## Session Logs

Every pipeline run automatically dumps a JSON file to `logs/` at the repo root. Each file is named by timestamp (`YYYY-MM-DD_HH-MM-SS.json`) and contains:

```json
{
  "config": { /* full NL2SQLPipelineConfig snapshot */ },
  "context": { /* complete context dict after pipeline execution */ }
}
```

This provides full traceability of every run — the exact config used and every intermediate value produced by the operators. The `logs/` directory is git-ignored.

---

## Docker

The project ships with a `Dockerfile` and `docker-compose.yml`. The image is built on `ghcr.io/astral-sh/uv:python3.14-bookworm-slim` (bundles uv + Python 3.14); no separate Python install is needed.

### Services

| Service | Description | Port |
|---|---|---|
| `ui` | Streamlit web UI | 8501 |
| `cli` | Interactive terminal REPL (requires `stdin_open + tty`) | — |
| `postgres` | PostgreSQL 17 (for adapter development and testing) | 5432 |

### Volumes

- `./logs:/app/logs` — session logs written by the pipeline persist on the host
- `./data:/app/data` — database file is bind-mounted, so you can swap the DB without rebuilding the image
- `postgres_data` (named) — PostgreSQL data persists across container restarts

### Secrets

Secrets (`OPENAI_API_KEY`, etc.) are passed at runtime via `env_file: .env`. They are never baked into the image. Copy `.env.example` → `.env` and fill in your key before running.

### Quick reference

```bash
docker compose build          # build the image
docker compose up ui          # web UI → http://localhost:8501
docker compose run --rm cli   # interactive CLI
docker compose up postgres -d # PostgreSQL only (background)
```

---

## Configuration Reference

`config.yaml` — all keys map 1:1 to Pydantic models in `pipeline/config.py`, validated at startup.

Three cross-cutting settings live directly under `nl2sql_pipeline`:

| Key | Where used | Notes |
|---|---|---|
| `db_file_path` | SchemaLinker + SQLExecutor | Single source of truth for the database location; injected into both operators at pipeline build time |
| `intent_guardrail.scope` | IntentGuardrail | Null skips the intent gate entirely |
| `schema_linker.schema_guardrails` | SchemaLinker | Null exposes the full schema; use `"*"` to allow all columns of a table |

All other keys are grouped per operator and are self-contained.
