from unittest.mock import MagicMock

import pytest

from vortosql.core.database import database_handler as dh_module
from vortosql.core.database.database_handler import DBMS, DatabaseHandler


@pytest.fixture
def mock_adapters(monkeypatch):
    sqlite_cls = MagicMock(name="SQLiteAdapterClass")
    duckdb_cls = MagicMock(name="DuckDBAdapterClass")
    sqlite_cls.return_value.connect.return_value = None
    duckdb_cls.return_value.connect.return_value = None
    monkeypatch.setattr(
        dh_module,
        "ADAPTERS",
        {DBMS.SQLITE: sqlite_cls, DBMS.DUCKDB: duckdb_cls},
    )
    return {"sqlite": sqlite_cls, "duckdb": duckdb_cls}


def test_handler_dispatches_to_sqlite_adapter(mock_adapters):
    params = {"db_path": "/tmp/x.db"}
    DatabaseHandler(DBMS.SQLITE, params)
    mock_adapters["sqlite"].assert_called_once_with(params)
    mock_adapters["duckdb"].assert_not_called()


def test_handler_dispatches_to_duckdb_adapter(mock_adapters):
    params = {"db_path": "/tmp/x.duckdb"}
    DatabaseHandler(DBMS.DUCKDB, params)
    mock_adapters["duckdb"].assert_called_once_with(params)
    mock_adapters["sqlite"].assert_not_called()


def test_handler_calls_connect_on_init(mock_adapters):
    DatabaseHandler(DBMS.SQLITE, {"db_path": ":memory:"})
    instance = mock_adapters["sqlite"].return_value
    instance.connect.assert_called_once_with()


def test_handler_unsupported_dbms_raises_value_error(monkeypatch):
    monkeypatch.setattr(dh_module, "ADAPTERS", {})
    with pytest.raises(ValueError, match="Unsupported database"):
        DatabaseHandler(DBMS.SQLITE, {"db_path": ":memory:"})


def test_handler_connect_failure_propagates(mock_adapters):
    mock_adapters["sqlite"].return_value.connect.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        DatabaseHandler(DBMS.SQLITE, {"db_path": ":memory:"})


def test_handler_run_query_delegates_to_adapter(mock_adapters):
    handler = DatabaseHandler(DBMS.SQLITE, {"db_path": ":memory:"})
    handler.run_query("SELECT 1")
    mock_adapters["sqlite"].return_value.run_query.assert_called_once_with(
        "SELECT 1", False
    )


def test_handler_run_query_passes_return_cursor_flag(mock_adapters):
    handler = DatabaseHandler(DBMS.SQLITE, {"db_path": ":memory:"})
    handler.run_query("SELECT 1", return_cursor=True)
    mock_adapters["sqlite"].return_value.run_query.assert_called_once_with(
        "SELECT 1", True
    )


def test_handler_close_connection_delegates(mock_adapters):
    handler = DatabaseHandler(DBMS.SQLITE, {"db_path": ":memory:"})
    handler.close_connection()
    mock_adapters["sqlite"].return_value.close_connection.assert_called_once_with()


def test_handler_is_connection_alive_true_when_query_succeeds(mock_adapters):
    mock_adapters["sqlite"].return_value.run_query.return_value = (["1"], [(1,)])
    handler = DatabaseHandler(DBMS.SQLITE, {"db_path": ":memory:"})
    assert handler.is_connection_alive() is True


def test_handler_is_connection_alive_false_when_query_raises(mock_adapters):
    mock_adapters["sqlite"].return_value.run_query.side_effect = RuntimeError("dead")
    handler = DatabaseHandler(DBMS.SQLITE, {"db_path": ":memory:"})
    assert handler.is_connection_alive() is False
