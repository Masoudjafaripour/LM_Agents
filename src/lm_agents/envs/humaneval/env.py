"""HumanEval as a multi-turn coding env: write a solution, get test feedback, retry."""

import functools
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from lm_agents.envs import register_env
from lm_agents.envs.base import Env, StepResult

PROMPT = """Complete the following Python function. Reply with the full function \
(including any imports it needs) in a single ```python code block.

```python
{prompt}```"""

FEEDBACK = """Your solution failed the tests:

```
{error}
```

Fix the function and reply with the full corrected version in a single ```python code block."""


@functools.lru_cache(maxsize=1)
def load_problems() -> dict[str, dict]:
    from datasets import load_dataset

    return {row["task_id"]: row for row in load_dataset("openai/openai_humaneval", split="test")}


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", text, re.DOTALL)
    return max(blocks, key=len) if blocks else text


def run_tests(program: str, timeout: float) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "solution.py"
        path.write_text(program)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(path)],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False, f"Timeout: tests did not finish within {timeout}s."
    return proc.returncode == 0, proc.stderr[-1500:]


@register_env("humaneval")
class HumanEvalEnv(Env):
    def __init__(self, max_turns: int = 3, timeout: float = 10.0):
        self.max_turns = max_turns
        self.timeout = timeout
        self.problems = load_problems()

    def task_ids(self) -> list[str]:
        return list(self.problems)

    def reset(self, task_id: str | None = None) -> str:
        self.problem = self.problems[task_id or next(iter(self.problems))]
        self.turn = 0
        return PROMPT.format(prompt=self.problem["prompt"])

    def step(self, action: str) -> StepResult:
        self.turn += 1
        p = self.problem
        # The prompt keeps the original imports; the model's code then redefines the function.
        program = "\n\n".join(
            [p["prompt"], extract_code(action), p["test"], f"check({p['entry_point']})\n"]
        )
        passed, error = run_tests(program, self.timeout)
        done = passed or self.turn >= self.max_turns
        return StepResult(
            observation="" if done else FEEDBACK.format(error=error.strip()),
            reward=float(passed),
            done=done,
            info={"turn": self.turn, "passed": passed},
        )
