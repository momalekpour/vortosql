from abc import ABC, abstractmethod
from typing import Any


class Operator(ABC):
    """Base class for pipeline operators.

    Operators share a mutable ``context`` dict — each reads its inputs from it
    and writes its outputs back, namespaced as ``<operator>_<key>`` (for
    example ``schema_linker_db_schema``, ``sql_executor_error``).

    To halt the pipeline early (out-of-scope question, fatal error, etc.),
    set ``context["pipeline_early_stop"]`` to a human-readable message;
    NL2SQLPipeline.execute breaks out of the operator loop as soon as it
    sees a truthy value there.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config

    @abstractmethod
    def execute(self, context: dict[str, Any]) -> None:
        pass
