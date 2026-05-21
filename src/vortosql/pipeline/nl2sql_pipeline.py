import json
import os
import time
from datetime import datetime
from typing import Any

from vortosql.core.logger import Logger
from vortosql.pipeline.answer_generator import AnswerGenerator
from vortosql.pipeline.config import NL2SQLPipelineConfig
from vortosql.pipeline.example_selector import ExampleSelector
from vortosql.pipeline.intent_guardrail import IntentGuardrail
from vortosql.pipeline.operator import Operator
from vortosql.pipeline.schema_linker import SchemaLinker
from vortosql.pipeline.sql_corrector import SQLCorrector
from vortosql.pipeline.sql_executor import SQLExecutor
from vortosql.pipeline.sql_generator import (
    SQLGenerationPromptTemplate,
    SQLGenerator,
)

logger = Logger(__name__)


class NL2SQLPipeline:
    def __init__(self, config: dict[str, Any]):
        self.config = NL2SQLPipelineConfig(**config)
        self.operators = self.build(self.config)

    @staticmethod
    def build(config: NL2SQLPipelineConfig) -> list[Operator]:
        operators = list()

        operators.append(IntentGuardrail(config=config.intent_guardrail.model_dump()))

        schema_linker_config = config.schema_linker.model_dump()
        schema_linker_config["db_file_path"] = config.db_file_path
        operators.append(SchemaLinker(config=schema_linker_config))

        if (
            config.sql_generator.prompt_template
            != SQLGenerationPromptTemplate.ZERO_SHOT
        ):
            operators.append(
                ExampleSelector(config=config.example_selector.model_dump())
            )
        operators.append(SQLGenerator(config=config.sql_generator.model_dump()))
        if config.sql_corrector.max_correction_attempts > 0:
            operators.append(SQLCorrector(config=config.sql_corrector.model_dump()))

        sql_executor_config = config.sql_executor.model_dump()
        sql_executor_config["db_file_path"] = config.db_file_path
        operators.append(SQLExecutor(config=sql_executor_config))

        operators.append(AnswerGenerator(config=config.answer_generator.model_dump()))

        return operators

    def execute(self, user_question: str) -> dict[str, Any]:
        context = {"user_question": user_question}
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        start_time = time.time()
        for operator in self.operators:
            operator.execute(context)
            if context.get("pipeline_early_stop"):
                break
        end_time = time.time()
        context["pipeline_latency"] = end_time - start_time
        context["timestamp"] = timestamp

        if os.environ.get("VORTOSQL_DUMP_SESSION_LOGS"):
            self._dump_session_log(context)

        return context

    def _dump_session_log(self, context: dict[str, Any]) -> None:
        logs_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        filename = context["timestamp"].replace(" ", "_").replace(":", "-") + ".json"
        filepath = os.path.join(logs_dir, filename)
        payload = {
            "config": self.config.model_dump(mode="json"),
            "context": context,
        }
        with open(filepath, "w") as f:
            json.dump(payload, f, indent=2, default=str)
