from lm_agents.data.rollout import rollout
from lm_agents.envs.base import Env, StepResult


class CountdownEnv(Env):
    max_turns = 5

    def reset(self, task_id=None):
        self.left = 3
        return "count down"

    def step(self, action):
        self.left -= 1
        return StepResult(
            observation=str(self.left), reward=float(self.left == 0), done=self.left == 0
        )


class EchoPolicy:
    def act(self, messages):
        return "next"


def test_rollout_terminates_and_collects_reward():
    traj = rollout(EchoPolicy(), CountdownEnv())
    assert traj.total_reward == 1.0
    assert [m["role"] for m in traj.messages].count("assistant") == 3
