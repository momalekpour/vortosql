from __future__ import annotations

from typing import Any

import pytest

from vortosql.core.model_manager import ModelProvider, OpenAIModel
from vortosql.pipeline.intent_guardrail import IntentGuardrail


@pytest.fixture
def base_config() -> dict[str, Any]:
    return {
        "scope": "questions about employees",
        "fail_closed": False,
        "chat_completion_model_provider": ModelProvider.OPENAI,
        "chat_completion_model_name": OpenAIModel.GPT_54_MINI,
        "temperature": 0,
    }


def test_skips_when_no_scope_configured(base_config, fake_llm_factory):
    from tests.pipeline.conftest import FakeChatCompletion

    fake_llm_factory(FakeChatCompletion([]))  # would explode if called
    base_config["scope"] = None
    guardrail = IntentGuardrail(base_config)

    ctx: dict[str, Any] = {"user_question": "anything"}
    guardrail.execute(ctx)

    assert ctx["intent_guardrail_is_in_scope"] is True
    assert "pipeline_early_stop" not in ctx


def test_in_scope_response_lets_pipeline_continue(base_config, fake_llm_factory):
    from tests.pipeline.conftest import FakeChatCompletion

    fake_llm_factory(FakeChatCompletion(['{"is_in_scope": true, "reason": "ok"}']))
    guardrail = IntentGuardrail(base_config)

    ctx: dict[str, Any] = {"user_question": "Who earns the most?"}
    guardrail.execute(ctx)

    assert ctx["intent_guardrail_is_in_scope"] is True
    assert "pipeline_early_stop" not in ctx


def test_out_of_scope_response_sets_early_stop(base_config, fake_llm_factory):
    from tests.pipeline.conftest import FakeChatCompletion

    fake_llm_factory(
        FakeChatCompletion(['{"is_in_scope": false, "reason": "off topic"}'])
    )
    guardrail = IntentGuardrail(base_config)

    ctx: dict[str, Any] = {"user_question": "Stock price?"}
    guardrail.execute(ctx)

    assert ctx["intent_guardrail_is_in_scope"] is False
    assert "off topic" in ctx["pipeline_early_stop"]


def test_strips_json_code_fence(base_config, fake_llm_factory):
    from tests.pipeline.conftest import FakeChatCompletion

    fenced = '```json\n{"is_in_scope": true, "reason": "fine"}\n```'
    fake_llm_factory(FakeChatCompletion([fenced]))
    guardrail = IntentGuardrail(base_config)

    ctx: dict[str, Any] = {"user_question": "Who's the CEO?"}
    guardrail.execute(ctx)

    assert ctx["intent_guardrail_is_in_scope"] is True
    assert ctx["intent_guardrail_reason"] == "fine"


def test_fail_open_on_llm_error(base_config, fake_llm_factory):
    from tests.pipeline.conftest import RaisingChatCompletion

    fake_llm_factory(RaisingChatCompletion(RuntimeError("network down")))
    base_config["fail_closed"] = False
    guardrail = IntentGuardrail(base_config)

    ctx: dict[str, Any] = {"user_question": "Who is highest paid?"}
    guardrail.execute(ctx)

    assert ctx["intent_guardrail_is_in_scope"] is True
    assert ctx["intent_guardrail_failed"] is True
    assert "pipeline_early_stop" not in ctx


def test_fail_closed_on_llm_error(base_config, fake_llm_factory):
    from tests.pipeline.conftest import RaisingChatCompletion

    fake_llm_factory(RaisingChatCompletion(RuntimeError("network down")))
    base_config["fail_closed"] = True
    guardrail = IntentGuardrail(base_config)

    ctx: dict[str, Any] = {"user_question": "Who is highest paid?"}
    guardrail.execute(ctx)

    assert ctx["intent_guardrail_is_in_scope"] is False
    assert ctx["intent_guardrail_failed"] is True
    assert "pipeline_early_stop" in ctx
