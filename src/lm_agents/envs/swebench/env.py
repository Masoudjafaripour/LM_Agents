from lm_agents.envs import register_env
from lm_agents.envs.base import Env, StepResult


@register_env("swebench")
class SWEBenchEnv(Env):
    """Agent edits a repo in a sandbox; reward = fraction of FAIL_TO_PASS tests that pass."""

    max_turns = 50

    def __init__(self, split: str = "lite"):
        self.split = split

    def reset(self, task_id: str | None = None) -> str:
        raise NotImplementedError("Load instance, spin up container, return issue text.")

    def step(self, action: str) -> StepResult:
        raise NotImplementedError("Execute tool call in container; on submit, run tests.")
