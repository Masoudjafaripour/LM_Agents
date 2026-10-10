"""Policy backed by any OpenAI-compatible server (e.g. `vllm serve`)."""

from openai import OpenAI

from lm_agents.agents.policy import Message


class OpenAIChatPolicy:
    def __init__(
        self,
        name: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.name = name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def act(self, messages: list[Message]) -> str:
        resp = self.client.chat.completions.create(
            model=self.name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content or ""
