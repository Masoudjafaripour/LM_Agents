from typing import Protocol

from lm_agents.data.rollout import Trajectory


class RewardFn(Protocol):
    def __call__(self, traj: Trajectory) -> float: ...


def task_success(traj: Trajectory) -> float:
    """Default: the env's own terminal reward."""
    return traj.rewards[-1] if traj.rewards else 0.0
