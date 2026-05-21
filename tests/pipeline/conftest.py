"""Shared fixtures for pipeline operator tests with mocked LLMs."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest


class FakeChatCompletion:
    """Drop-in for OpenAIChatCompletion that returns canned content."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def get_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("FakeChatCompletion ran out of canned responses")
        content = self._responses.pop(0)
        return {
            "completion_content": [content],
            "completion_latency": 0.01,
            "num_input_tokens": 1,
            "num_output_tokens": 1,
        }


class RaisingChatCompletion:
    """LLM that always raises — used to exercise error paths."""

    def __init__(self, exc: Exception):
        self._exc = exc

    def get_chat_completion(self, **kwargs):
        raise self._exc


@pytest.fixture
def fake_llm_factory(monkeypatch):
    """Patches ModelManager.create_model to return whatever you stash."""
    from vortosql.core.model_manager import model_manager as mm

    holder: dict[str, Any] = {"llm": None}

    def _set(llm):
        holder["llm"] = llm

        def _factory(*args, **kwargs):
            return holder["llm"]

        monkeypatch.setattr(mm.ModelManager, "create_model", _factory)
        return llm

    return _set


@pytest.fixture
def schema_db(tmp_path: Path) -> Path:
    """Tiny multi-table SQLite DB for SchemaLinker tests."""
    db_path = tmp_path / "schema.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript("""
            CREATE TABLE Employee (
                EmployeeId INTEGER PRIMARY KEY,
                Name       TEXT NOT NULL,
                SalaryAmount REAL,
                Role       TEXT,
                ManagerId  INTEGER REFERENCES Employee(EmployeeId)
            );
            CREATE TABLE Certification (
                CertificationId INTEGER PRIMARY KEY,
                EmployeeId      INTEGER REFERENCES Employee(EmployeeId),
                CertName        TEXT
            );
            INSERT INTO Employee VALUES
                (1, 'Alice', 100.0, 'eng', NULL),
                (2, 'Bob',    90.0, 'eng', 1);
            INSERT INTO Certification VALUES (1, 1, 'AWS');
            """)
        conn.commit()
    finally:
        conn.close()
    return db_path
