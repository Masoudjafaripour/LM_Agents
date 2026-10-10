from abc import ABC, abstractmethod
from typing import Any


class Trainer(ABC):
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg

    @abstractmethod
    def train(self) -> None: ...
