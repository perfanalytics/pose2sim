from abc import ABC, abstractmethod
from typing import Any

class BaseStage(ABC):
    name: str = "base"

    def __init__(self, config, **kwargs) -> None:
        self.config = config

    def setup(self) -> None:
        pass

    @abstractmethod
    def run(self, data_in: Any) -> Any:
        ...

    @abstractmethod
    def save_data(self, data_out: Any) -> None:
        ...

    def teardown(self) -> None:
        pass