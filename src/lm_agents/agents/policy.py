"""Policy wrapper: turns a chat history into the next action."""

from typing import Protocol

Message = dict[str, str]  # {"role": ..., "content": ...}


class Policy(Protocol):
    def act(self, messages: list[Message]) -> str: ...
