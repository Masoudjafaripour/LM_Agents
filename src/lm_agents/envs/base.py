"""Common interface every benchmark env implements, so one rollout loop serves all trainers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepResult:
    observation: str
    reward: float = 0.0
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class Env(ABC):
    max_turns: int = 30

    @abstractmethod
    def reset(self, task_id: str | None = None) -> str:
        """Start a new episode and return the initial observation (task prompt)."""

    @abstractmethod
    def step(self, action: str) -> StepResult:
        """Apply the agent's action (text / tool call) and return the result."""

    def task_ids(self) -> list[str]:
        """All task ids in the configured split."""
        raise NotImplementedError

    def close(self) -> None:
        pass
