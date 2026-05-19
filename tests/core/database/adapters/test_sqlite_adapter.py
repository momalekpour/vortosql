import sqlite3
from pathlib import Path

import pytest

from vortosql.core.database.adapters.sqlite_adapter import SQLiteAdapter


def test_connect_creates_db_file(sqlite_db_path: Path):
    assert not sqlite_db_path.exists()
    adapter = SQLiteAdapter({"db_path": str(sqlite_db_path)})
    adapter.connect()
    try:
        assert sqlite_db_path.exists()
    finally:
        adapter.close_connection()


def test_connect_sets_connection_attribute(sqlite_db_path: Path):
    adapter = SQLiteAdapter({"db_path": str(sqlite_db_path)})
    assert adapter.connection is None
    adapter.connect()
    try:
        assert isinstance(adapter.connection, sqlite3.Connection)
    finally:
        adapter.close_connection()


def test_run_query_returns_columns_and_rows(seeded_sqlite: Path):
    adapter = SQLiteAdapter({"db_path": str(seeded_sqlite)})
    adapter.connect()
    try:
        columns, rows = adapter.run_query("SELECT id, name FROM users ORDER BY id")
    finally:
        adapter.close_connection()

    assert columns == ["id", "name"]
    assert rows == [(1, "alice"), (2, "bob"), (3, "carol")]


def test_run_query_empty_result(seeded_sqlite: Path):
    adapter = SQLiteAdapter({"db_path": str(seeded_sqlite)})
    adapter.connect()
    try:
        columns, rows = adapter.run_query("SELECT id, name FROM users WHERE id = 999")
    finally:
        adapter.close_connection()

    assert columns == ["id", "name"]
    assert rows == []


def test_run_query_multiple_columns(seeded_sqlite: Path):
    adapter = SQLiteAdapter({"db_path": str(seeded_sqlite)})
    adapter.connect()
    try:
        columns, rows = adapter.run_query(
            "SELECT id, name, id * 2 AS doubled FROM users ORDER BY id"
        )
    finally:
        adapter.close_connection()

    assert columns == ["id", "name", "doubled"]
    assert rows == [(1, "alice", 2), (2, "bob", 4), (3, "carol", 6)]


def test_run_query_return_cursor_true(seeded_sqlite: Path):
    adapter = SQLiteAdapter({"db_path": str(seeded_sqlite)})
    adapter.connect()
    try:
        cursor = adapter.run_query(
            "SELECT id FROM users ORDER BY id", return_cursor=True
        )
        assert isinstance(cursor, sqlite3.Cursor)
        assert cursor.fetchall() == [(1,), (2,), (3,)]
    finally:
        adapter.close_connection()


def test_run_query_invalid_sql_raises(seeded_sqlite: Path):
    adapter = SQLiteAdapter({"db_path": str(seeded_sqlite)})
    adapter.connect()
    try:
        with pytest.raises(sqlite3.OperationalError):
            adapter.run_query("SELEKT bogus FROM nowhere")
    finally:
        adapter.close_connection()


def test_close_connection_closes(sqlite_db_path: Path):
    adapter = SQLiteAdapter({"db_path": str(sqlite_db_path)})
    adapter.connect()
    connection = adapter.connection
    adapter.close_connection()
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")


def test_close_connection_when_never_connected_does_not_raise(sqlite_db_path: Path):
    adapter = SQLiteAdapter({"db_path": str(sqlite_db_path)})
    adapter.close_connection()


def test_reconnect_after_close(seeded_sqlite: Path):
    adapter = SQLiteAdapter({"db_path": str(seeded_sqlite)})
    adapter.connect()
    adapter.close_connection()
    adapter.connect()
    try:
        columns, rows = adapter.run_query("SELECT COUNT(*) AS n FROM users")
        assert columns == ["n"]
        assert rows == [(3,)]
    finally:
        adapter.close_connection()
