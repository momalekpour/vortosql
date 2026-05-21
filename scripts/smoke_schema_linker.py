"""Ad-hoc smoke test for SchemaLinker techniques.

Run from the repo root with `OPENAI_API_KEY` set in your environment:

    uv run python scripts/smoke_schema_linker.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vortosql.core.model_manager import ModelProvider, OpenAIModel
from vortosql.pipeline.schema_linker import SchemaLinker, SchemaLinkingTechnique


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    db_file_path = str(repo_root / "data" / "employees.db")
    question = "What is the name of the employee with the highest salary?"

    schema_guardrails = {
        "Employee": ["EmployeeId", "Name", "SalaryAmount", "Role"],
        "Certification": ["CertificationId"],
    }

    print("--------\nFull Schema (unrestricted)")
    linker = SchemaLinker(
        config={
            "db_file_path": db_file_path,
            "technique": SchemaLinkingTechnique.FULL,
        }
    )
    ctx: dict[str, Any] = {}
    linker.execute(ctx)
    print(ctx["schema_linker_db_schema"])

    print("--------\nFull Schema (restricted)")
    linker = SchemaLinker(
        config={
            "db_file_path": db_file_path,
            "technique": SchemaLinkingTechnique.FULL,
            "schema_guardrails": schema_guardrails,
        }
    )
    ctx = {}
    linker.execute(ctx)
    print(ctx["schema_linker_db_schema"])
    print("Columns by table:", ctx["schema_linker_db_columns"])

    print("--------\nTCSL (restricted)")
    linker = SchemaLinker(
        config={
            "db_file_path": db_file_path,
            "technique": SchemaLinkingTechnique.TCSL,
            "model_provider": ModelProvider.OPENAI,
            "model_name": OpenAIModel.GPT_54_MINI,
            "schema_guardrails": schema_guardrails,
        }
    )
    ctx = {"user_question": question}
    linker.execute(ctx)
    print(ctx["schema_linker_db_schema"])
    print("Columns by table:", ctx["schema_linker_db_columns"])

    print("--------\nSCSL (restricted)")
    linker = SchemaLinker(
        config={
            "db_file_path": db_file_path,
            "technique": SchemaLinkingTechnique.SCSL,
            "model_provider": ModelProvider.OPENAI,
            "model_name": OpenAIModel.GPT_54_MINI,
            "schema_guardrails": schema_guardrails,
        }
    )
    ctx = {"user_question": question}
    linker.execute(ctx)
    print(ctx["schema_linker_db_schema"])
    print("Columns by table:", ctx["schema_linker_db_columns"])


if __name__ == "__main__":
    main()
