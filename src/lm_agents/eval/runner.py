import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from lm_agents.agents.policy import Policy
from lm_agents.data.rollout import Trajectory, rollout
from lm_agents.envs.base import Env


def evaluate(
    policy: Policy,
    env_fn: Callable[[], Env],
    task_ids: list[str],
    workers: int = 1,
    out_dir: str | None = None,
) -> dict[str, float]:
    def run_one(task_id: str) -> Trajectory:
        env = env_fn()  # one env per episode, so episodes can run in parallel
        try:
            return rollout(policy, env, task_id)
        finally:
            env.close()

    with ThreadPoolExecutor(workers) as pool:
        trajs = list(pool.map(run_one, task_ids))

    # Turn (1-based) at which each task was first solved, or None.
    solved_at = [next((i + 1 for i, r in enumerate(t.rewards) if r > 0), None) for t in trajs]
    n = max(len(trajs), 1)
    metrics = {"n": len(trajs), "success_rate": sum(s is not None for s in solved_at) / n}
    for k in range(1, max((len(t.rewards) for t in trajs), default=0) + 1):
        metrics[f"solved_by_turn_{k}"] = sum(s is not None and s <= k for s in solved_at) / n

    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "trajectories.jsonl", "w") as f:
            f.writelines(
                json.dumps({"task_id": task_id, "solved_at": s, **asdict(traj)}) + "\n"
                for task_id, traj, s in zip(task_ids, trajs, solved_at)
            )
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
