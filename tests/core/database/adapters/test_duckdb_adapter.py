from pathlib import Path

import duckdb
import pytest

from vortosql.core.database.adapters.duckdb_adapter import DuckDBAdapter


def test_connect_creates_db_file(duckdb_db_path: Path):
    assert not duckdb_db_path.exists()
    adapter = DuckDBAdapter({"db_path": str(duckdb_db_path)})
    adapter.connect()
    try:
        assert duckdb_db_path.exists()
    finally:
        adapter.close_connection()


def test_connect_sets_connection_attribute(duckdb_db_path: Path):
    adapter = DuckDBAdapter({"db_path": str(duckdb_db_path)})
    assert adapter.connection is None
    adapter.connect()
    try:
        assert isinstance(adapter.connection, duckdb.DuckDBPyConnection)
    finally:
        adapter.close_connection()


def test_run_query_returns_columns_and_rows(seeded_duckdb: Path):
    adapter = DuckDBAdapter({"db_path": str(seeded_duckdb)})
    adapter.connect()
    try:
        columns, rows = adapter.run_query("SELECT id, name FROM users ORDER BY id")
    finally:
        adapter.close_connection()

    assert columns == ["id", "name"]
    assert rows == [(1, "alice"), (2, "bob"), (3, "carol")]


def test_run_query_empty_result(seeded_duckdb: Path):
    adapter = DuckDBAdapter({"db_path": str(seeded_duckdb)})
    adapter.connect()
    try:
        columns, rows = adapter.run_query("SELECT id, name FROM users WHERE id = 999")
    finally:
        adapter.close_connection()

    assert columns == ["id", "name"]
    assert rows == []


def test_run_query_multiple_columns(seeded_duckdb: Path):
    adapter = DuckDBAdapter({"db_path": str(seeded_duckdb)})
    adapter.connect()
    try:
        columns, rows = adapter.run_query(
            "SELECT id, name, id * 2 AS doubled FROM users ORDER BY id"
        )
    finally:
        adapter.close_connection()

    assert columns == ["id", "name", "doubled"]
    assert rows == [(1, "alice", 2), (2, "bob", 4), (3, "carol", 6)]


def test_run_query_return_cursor_true(seeded_duckdb: Path):
    adapter = DuckDBAdapter({"db_path": str(seeded_duckdb)})
    adapter.connect()
    try:
        cursor = adapter.run_query(
            "SELECT id FROM users ORDER BY id", return_cursor=True
        )
        assert isinstance(cursor, duckdb.DuckDBPyConnection)
        assert cursor.fetchall() == [(1,), (2,), (3,)]
    finally:
        adapter.close_connection()


def test_run_query_invalid_sql_raises(seeded_duckdb: Path):
    adapter = DuckDBAdapter({"db_path": str(seeded_duckdb)})
    adapter.connect()
    try:
        with pytest.raises(duckdb.Error):
            adapter.run_query("SELEKT bogus FROM nowhere")
    finally:
        adapter.close_connection()


def test_close_connection_closes(duckdb_db_path: Path):
    adapter = DuckDBAdapter({"db_path": str(duckdb_db_path)})
    adapter.connect()
    connection = adapter.connection
    adapter.close_connection()
    with pytest.raises(duckdb.ConnectionException):
        connection.execute("SELECT 1")


def test_close_connection_when_never_connected_does_not_raise(duckdb_db_path: Path):
    adapter = DuckDBAdapter({"db_path": str(duckdb_db_path)})
    adapter.close_connection()


def test_reconnect_after_close(seeded_duckdb: Path):
    adapter = DuckDBAdapter({"db_path": str(seeded_duckdb)})
    adapter.connect()
    adapter.close_connection()
    adapter.connect()
    try:
        columns, rows = adapter.run_query("SELECT COUNT(*) AS n FROM users")
        assert columns == ["n"]
        assert rows == [(3,)]
    finally:
        adapter.close_connection()
