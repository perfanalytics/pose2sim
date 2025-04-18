from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseStage(ABC):
    name: str = "base"
    stream: bool = False
    sentinel = object()

    def setup(self) -> None:
        pass

    @abstractmethod
    def run(self, *args, **kwargs): ...

    @abstractmethod
    def save_data(self, data_out: Any) -> None:
        ...

    def teardown(self) -> None:
        pass