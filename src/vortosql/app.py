from pathlib import Path
from typing import Any

import yaml

from vortosql.core.logger import Logger
from vortosql.pipeline.nl2sql_pipeline import NL2SQLPipeline

logger = Logger(__name__)


class NL2SQLApp:
    def __init__(
        self,
        config_path: str = "config.yaml",
        scope: str | None = None,
        schema_guardrails: dict[str, list[str]] | None = None,
    ):
        config_file = Path(config_path).resolve()
        with open(config_file) as f:
            config = yaml.safe_load(f)["nl2sql_pipeline"]

        db_path = Path(config["db_file_path"])
        if not db_path.is_absolute():
            db_path = (config_file.parent / db_path).resolve()
        config["db_file_path"] = str(db_path)

        if scope is not None:
            config["intent_guardrail"]["scope"] = scope
        if schema_guardrails is not None:
            config["schema_linker"]["schema_guardrails"] = schema_guardrails

        self._pipeline = NL2SQLPipeline(config=config)

    def ask(self, user_question: str) -> dict[str, Any]:
        return self._pipeline.execute(user_question)
