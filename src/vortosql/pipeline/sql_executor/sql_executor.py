from typing import Any

from vortosql.core.database.database_handler import DatabaseHandler
from vortosql.core.logger import Logger
from vortosql.pipeline.operator import Operator

logger = Logger(__name__)


class SQLExecutor(Operator):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)

    def execute(self, context: dict[str, Any]) -> None:
        sql_query = context.get("sql_query", "")
        try:
            db = DatabaseHandler(
                self.config["dbms"], {"db_path": self.config["db_file_path"]}
            )
            try:
                columns, rows = db.run_query(sql_query)
            finally:
                db.close_connection()

            context["sql_executor_sql_query"] = sql_query
            context["sql_executor_columns"] = list(columns)
            context["sql_executor_rows"] = [list(r) for r in rows]
            context["sql_executor_row_count"] = len(rows)
            context["sql_executor_error"] = None

            logger.log(
                "info",
                "SQL_EXECUTED_SUCCESSFULLY",
                {"row_count": len(rows), "sql_query": sql_query},
            )
        except Exception as e:
            logger.log("error", "ERROR_IN_SQL_EXECUTOR_OPERATOR", {"error": str(e)})
            context["sql_executor_sql_query"] = sql_query
            context["sql_executor_columns"] = []
            context["sql_executor_rows"] = []
            context["sql_executor_row_count"] = 0
            context["sql_executor_error"] = str(e)
