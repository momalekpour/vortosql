from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from vortosql.pipeline.schema_linker import SchemaLinker, SchemaLinkingTechnique


@pytest.fixture
def linker(schema_db: Path) -> SchemaLinker:
    return SchemaLinker(
        config={
            "db_file_path": str(schema_db),
            "technique": SchemaLinkingTechnique.FULL,
        }
    )


def test_full_schema_lists_all_tables(linker: SchemaLinker):
    ctx: dict[str, Any] = {}
    linker.execute(ctx)

    schema = ctx["schema_linker_db_schema"]
    assert "Employee" in schema
    assert "Certification" in schema
    assert set(ctx["schema_linker_db_columns"]) == {"Employee", "Certification"}


def test_guardrails_filter_columns(schema_db: Path):
    linker = SchemaLinker(
        config={
            "db_file_path": str(schema_db),
            "technique": SchemaLinkingTechnique.FULL,
            "schema_guardrails": {"Employee": ["EmployeeId", "Name"]},
        }
    )
    ctx: dict[str, Any] = {}
    linker.execute(ctx)

    cols = ctx["schema_linker_db_columns"]
    assert set(cols.keys()) == {"Employee"}
    assert set(cols["Employee"]) == {"EmployeeId", "Name"}

    schema = ctx["schema_linker_db_schema"]
    assert "SalaryAmount" not in schema
    assert "Certification" not in schema


def test_star_allows_all_columns(schema_db: Path):
    linker = SchemaLinker(
        config={
            "db_file_path": str(schema_db),
            "technique": SchemaLinkingTechnique.FULL,
            "schema_guardrails": {"Employee": ["*"]},
        }
    )
    ctx: dict[str, Any] = {}
    linker.execute(ctx)

    cols = ctx["schema_linker_db_columns"]
    assert set(cols["Employee"]) == {
        "EmployeeId",
        "Name",
        "SalaryAmount",
        "Role",
        "ManagerId",
    }


def test_guardrails_do_not_mutate_self_tables(schema_db: Path):
    linker = SchemaLinker(
        config={
            "db_file_path": str(schema_db),
            "technique": SchemaLinkingTechnique.FULL,
            "schema_guardrails": {"Employee": ["EmployeeId"]},
        }
    )
    original_cols = [c.column_name for c in linker.tables[0].columns]

    ctx: dict[str, Any] = {}
    linker.execute(ctx)

    after_cols = [c.column_name for c in linker.tables[0].columns]
    assert original_cols == after_cols  # internal state untouched


def test_missing_db_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        SchemaLinker(
            config={
                "db_file_path": str(tmp_path / "nope.db"),
                "technique": SchemaLinkingTechnique.FULL,
            }
        )
