from abc import ABC, abstractmethod
from typing import Dict, Any


class Operator(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> None:
        pass
