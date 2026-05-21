"""Ad-hoc smoke test for the full NL2SQL pipeline.

Run from the repo root with `OPENAI_API_KEY` set in your environment:

    uv run python scripts/smoke_pipeline.py
"""

from __future__ import annotations

from pathlib import Path

import yaml

from vortosql.pipeline.nl2sql_pipeline import NL2SQLPipeline


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    with open(repo_root / "config.yaml") as f:
        config = yaml.safe_load(f)["nl2sql_pipeline"]

    db_path = Path(config["db_file_path"])
    if not db_path.is_absolute():
        config["db_file_path"] = str((repo_root / db_path).resolve())

    pipeline = NL2SQLPipeline(config=config)

    result = pipeline.execute(
        user_question="What is the name of the employee with the highest salary?",
    )

    print(f"Generated SQL : {result.get('sql_generator_sql_query')}")
    print(f"Executed SQL  : {result.get('sql_executor_sql_query')}")
    print(f"Rows          : {result.get('sql_executor_rows')}")
    print(f"Latency       : {result['pipeline_latency']:.2f}s")


if __name__ == "__main__":
    main()
