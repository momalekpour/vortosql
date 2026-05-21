from __future__ import annotations

from typing import Any

import pytest

from vortosql.core.database import DBMS
from vortosql.core.model_manager import ModelProvider, OpenAIModel
from vortosql.pipeline.sql_corrector import SQLCorrectionPromptTemplate, SQLCorrector


@pytest.fixture
def base_config() -> dict[str, Any]:
    return {
        "prompt_template": SQLCorrectionPromptTemplate.SYNTAX_CORRECTION,
        "max_correction_attempts": 2,
        "dbms": DBMS.SQLITE,
        "chat_completion_model_provider": ModelProvider.OPENAI,
        "chat_completion_model_name": OpenAIModel.GPT_54_MINI,
        "temperature": 0,
        "random_seed": None,
    }


def test_parsable_query_skips_llm(base_config, fake_llm_factory):
    from tests.pipeline.conftest import FakeChatCompletion

    fake = fake_llm_factory(FakeChatCompletion([]))  # would error if called
    corrector = SQLCorrector(base_config)

    ctx: dict[str, Any] = {
        "schema_linker_db_schema": "Table 'x': (a INT)",
        "user_question": "q",
        "sql_query": "SELECT a FROM x",
    }
    corrector.execute(ctx)

    assert ctx["sql_corrector_is_successful"] is True
    assert ctx["sql_corrector_num_attempts"] == 0
    assert fake.calls == []
    assert "pipeline_early_stop" not in ctx


def test_unparsable_then_fixed(base_config, fake_llm_factory):
    from tests.pipeline.conftest import FakeChatCompletion

    fake_llm_factory(FakeChatCompletion(["SELECT a FROM x"]))
    corrector = SQLCorrector(base_config)

    ctx: dict[str, Any] = {
        "schema_linker_db_schema": "Table 'x': (a INT)",
        "user_question": "q",
        "sql_query": "SELEC a FROMM x",
    }
    corrector.execute(ctx)

    assert ctx["sql_corrector_is_successful"] is True
    assert ctx["sql_query"] == "SELECT a FROM x"
    assert "pipeline_early_stop" not in ctx


def test_exhaustion_sets_early_stop(base_config, fake_llm_factory):
    from tests.pipeline.conftest import FakeChatCompletion

    fake_llm_factory(FakeChatCompletion(["SELECT (", "FROM )"]))
    corrector = SQLCorrector(base_config)

    ctx: dict[str, Any] = {
        "schema_linker_db_schema": "Table 'x': (a INT)",
        "user_question": "q",
        "sql_query": "SELECT (",
    }
    corrector.execute(ctx)

    assert ctx["sql_corrector_is_successful"] is False
    assert ctx["sql_corrector_num_attempts"] == 2
    assert "pipeline_early_stop" in ctx
    assert "correction failed" in ctx["pipeline_early_stop"].lower()


def test_zero_attempts_disables_correction(base_config, fake_llm_factory):
    from tests.pipeline.conftest import FakeChatCompletion

    fake = fake_llm_factory(FakeChatCompletion([]))
    base_config["max_correction_attempts"] = 0
    corrector = SQLCorrector(base_config)

    ctx: dict[str, Any] = {
        "schema_linker_db_schema": "Table 'x': (a INT)",
        "user_question": "q",
        "sql_query": "SELEC a FROMM x",
    }
    corrector.execute(ctx)

    assert ctx["sql_corrector_is_successful"] is False
    assert ctx["sql_corrector_num_attempts"] == 0
    assert fake.calls == []
    assert "pipeline_early_stop" not in ctx
