from typing import Any

import psycopg2

from vortosql.core.database.adapters.base_adapter import BaseAdapter
from vortosql.core.logger import Logger

logger = Logger(__name__)


class PostgresAdapter(BaseAdapter):
    def __init__(self, connection_params: dict[str, Any]):
        """
        connection_params:
        {
            "host": "localhost",
            "port": 5432,
            "dbname": "vortosql",
            "user": "vortosql",
            "password": "vortosql",
        }
        """
        self.connection = None
        self.connection_params = connection_params

    def connect(self) -> None:
        try:
            self.connection = psycopg2.connect(**self.connection_params)
            logger.log(
                "info",
                "CONNECTION_ESTABLISHED",
                {
                    "host": self.connection_params.get("host"),
                    "dbname": self.connection_params.get("dbname"),
                },
            )
        except Exception as e:
            logger.log("error", "CONNECTION_FAILED", {"error": str(e)})
            raise

    def run_query(
        self, query: str, return_cursor: bool = False
    ) -> tuple[list[str], list[Any]] | psycopg2.extensions.cursor:
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            logger.log("info", "QUERY_EXECUTED", {"query": " ".join(query.split())})

            if return_cursor:
                return cursor

            column_names = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            cursor.close()
            return column_names, rows
        except Exception as e:
            logger.log(
                "error",
                "QUERY_EXECUTION_FAILED",
                {"query": " ".join(query.split()), "error": str(e)},
            )
            raise

    def close_connection(self) -> None:
        if self.connection:
            self.connection.close()
            logger.log(
                "info",
                "CONNECTION_CLOSED",
                {"host": self.connection_params.get("host")},
            )
        else:
            logger.log("warning", "CLOSE_ATTEMPTED_WITH_NO_CONNECTION")
