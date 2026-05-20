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
        with open(config_path) as f:
            config = yaml.safe_load(f)["nl2sql_pipeline"]

        if scope is not None:
            config["intent_guardrail"]["scope"] = scope
        if schema_guardrails is not None:
            config["schema_linker"]["schema_guardrails"] = schema_guardrails

        self._pipeline = NL2SQLPipeline(config=config)

    def ask(self, user_question: str) -> dict[str, Any]:
        return self._pipeline.execute(user_question)
