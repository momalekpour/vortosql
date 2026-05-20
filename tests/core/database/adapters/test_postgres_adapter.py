import os
from uuid import uuid4

import pytest

from vortosql.core.database.adapters.postgres_adapter import PostgresAdapter

psycopg2 = pytest.importorskip("psycopg2")


def _pg_params() -> dict:
    return {
        "host": os.environ.get("POSTGRES_HOST", "localhost"),
        "port": int(os.environ.get("POSTGRES_PORT", "5432")),
        "dbname": os.environ.get("POSTGRES_DB", "vortosql"),
        "user": os.environ.get("POSTGRES_USER", "vortosql"),
        "password": os.environ.get("POSTGRES_PASSWORD", "vortosql"),
    }


@pytest.fixture(scope="module")
def pg_params() -> dict:
    params = _pg_params()
    try:
        conn = psycopg2.connect(**params)
        conn.close()
    except Exception:
        pytest.skip(
            "PostgreSQL not reachable — start via `docker compose up postgres -d`"
        )
    return params


@pytest.fixture
def seeded_postgres(pg_params: dict):
    table = f"test_users_{uuid4().hex[:8]}"
    conn = psycopg2.connect(**pg_params)
    cur = conn.cursor()
    cur.execute(f'CREATE TABLE "{table}" (id INTEGER, name TEXT)')
    cur.executemany(
        f'INSERT INTO "{table}" VALUES (%s, %s)',
        [(1, "alice"), (2, "bob"), (3, "carol")],
    )
    conn.commit()
    cur.close()
    conn.close()

    yield {**pg_params, "_table": table}

    conn = psycopg2.connect(**pg_params)
    cur = conn.cursor()
    cur.execute(f'DROP TABLE IF EXISTS "{table}"')
    conn.commit()
    cur.close()
    conn.close()


def _adapter_params(seeded: dict) -> tuple[dict, str]:
    table = seeded["_table"]
    params = {k: v for k, v in seeded.items() if k != "_table"}
    return params, table


def test_connect_sets_connection_attribute(pg_params: dict):
    adapter = PostgresAdapter(pg_params)
    assert adapter.connection is None
    adapter.connect()
    try:
        assert adapter.connection is not None
    finally:
        adapter.close_connection()


def test_run_query_returns_columns_and_rows(seeded_postgres: dict):
    params, table = _adapter_params(seeded_postgres)
    adapter = PostgresAdapter(params)
    adapter.connect()
    try:
        columns, rows = adapter.run_query(f'SELECT id, name FROM "{table}" ORDER BY id')
    finally:
        adapter.close_connection()

    assert columns == ["id", "name"]
    assert rows == [(1, "alice"), (2, "bob"), (3, "carol")]


def test_run_query_empty_result(seeded_postgres: dict):
    params, table = _adapter_params(seeded_postgres)
    adapter = PostgresAdapter(params)
    adapter.connect()
    try:
        columns, rows = adapter.run_query(
            f'SELECT id, name FROM "{table}" WHERE id = 999'
        )
    finally:
        adapter.close_connection()

    assert columns == ["id", "name"]
    assert rows == []


def test_run_query_multiple_columns(seeded_postgres: dict):
    params, table = _adapter_params(seeded_postgres)
    adapter = PostgresAdapter(params)
    adapter.connect()
    try:
        columns, rows = adapter.run_query(
            f'SELECT id, name, id * 2 AS doubled FROM "{table}" ORDER BY id'
        )
    finally:
        adapter.close_connection()

    assert columns == ["id", "name", "doubled"]
    assert rows == [(1, "alice", 2), (2, "bob", 4), (3, "carol", 6)]


def test_run_query_return_cursor_true(seeded_postgres: dict):
    params, table = _adapter_params(seeded_postgres)
    adapter = PostgresAdapter(params)
    adapter.connect()
    try:
        cursor = adapter.run_query(
            f'SELECT id FROM "{table}" ORDER BY id', return_cursor=True
        )
        assert cursor.fetchall() == [(1,), (2,), (3,)]
    finally:
        adapter.close_connection()


def test_run_query_invalid_sql_raises(pg_params: dict):
    adapter = PostgresAdapter(pg_params)
    adapter.connect()
    try:
        with pytest.raises(Exception):
            adapter.run_query("SELEKT bogus FROM nowhere")
    finally:
        adapter.close_connection()


def test_close_connection_closes(pg_params: dict):
    adapter = PostgresAdapter(pg_params)
    adapter.connect()
    connection = adapter.connection
    adapter.close_connection()
    with pytest.raises(psycopg2.InterfaceError):
        connection.cursor()


def test_close_connection_when_never_connected_does_not_raise(pg_params: dict):
    adapter = PostgresAdapter(pg_params)
    adapter.close_connection()


def test_reconnect_after_close(seeded_postgres: dict):
    params, table = _adapter_params(seeded_postgres)
    adapter = PostgresAdapter(params)
    adapter.connect()
    adapter.close_connection()
    adapter.connect()
    try:
        columns, rows = adapter.run_query(f'SELECT COUNT(*) AS n FROM "{table}"')
        assert columns == ["n"]
        assert rows == [(3,)]
    finally:
        adapter.close_connection()
