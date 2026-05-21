from __future__ import annotations

from pathlib import Path
from typing import Any

from vortosql.core.database import DBMS
from vortosql.pipeline.sql_executor import SQLExecutor


def _config(db_path: Path) -> dict[str, Any]:
    return {"dbms": DBMS.SQLITE, "db_file_path": str(db_path)}


def test_runs_valid_query(seeded_sqlite: Path):
    executor = SQLExecutor(_config(seeded_sqlite))
    ctx: dict[str, Any] = {"sql_query": "SELECT id, name FROM users ORDER BY id"}
    executor.execute(ctx)

    assert ctx["sql_executor_error"] is None
    assert ctx["sql_executor_row_count"] == 3
    assert ctx["sql_executor_columns"] == ["id", "name"]
    assert ctx["sql_executor_rows"][0] == [1, "alice"]
    assert "pipeline_early_stop" not in ctx


def test_db_error_sets_early_stop(seeded_sqlite: Path):
    executor = SQLExecutor(_config(seeded_sqlite))
    ctx: dict[str, Any] = {"sql_query": "SELECT no_such_col FROM users"}
    executor.execute(ctx)

    assert ctx["sql_executor_error"]
    assert ctx["sql_executor_row_count"] == 0
    assert ctx["sql_executor_columns"] == []
    assert "pipeline_early_stop" in ctx
    assert "SQL execution failed" in ctx["pipeline_early_stop"]
