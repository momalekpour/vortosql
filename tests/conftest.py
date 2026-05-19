import os

os.environ.setdefault("LOG_LEVEL", "CRITICAL")

import sqlite3
from pathlib import Path

import duckdb
import pytest


@pytest.fixture
def sqlite_db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture
def duckdb_db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.duckdb"


@pytest.fixture
def seeded_sqlite(sqlite_db_path: Path) -> Path:
    conn = sqlite3.connect(sqlite_db_path)
    try:
        conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
        conn.executemany(
            "INSERT INTO users VALUES (?, ?)",
            [(1, "alice"), (2, "bob"), (3, "carol")],
        )
        conn.commit()
    finally:
        conn.close()
    return sqlite_db_path


@pytest.fixture
def seeded_duckdb(duckdb_db_path: Path) -> Path:
    conn = duckdb.connect(str(duckdb_db_path))
    try:
        conn.execute("CREATE TABLE users (id INTEGER, name VARCHAR)")
        conn.executemany(
            "INSERT INTO users VALUES (?, ?)",
            [(1, "alice"), (2, "bob"), (3, "carol")],
        )
    finally:
        conn.close()
    return duckdb_db_path
