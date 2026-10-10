from lm_agents.agents.policy import Policy
from lm_agents.data.rollout import rollout
from lm_agents.envs.base import Env


def evaluate(policy: Policy, env: Env, task_ids: list[str]) -> dict[str, float]:
    rewards = [rollout(policy, env, t).total_reward for t in task_ids]
    n = max(len(rewards), 1)
    return {
        "n": len(rewards),
        "mean_reward": sum(rewards) / n,
        "success_rate": sum(r > 0 for r in rewards) / n,
    }
