"""Usage: python scripts/eval.py --config configs/eval/humaneval.yaml [--limit N]"""

import argparse
import json
from pathlib import Path

import yaml
from plot import make_plots

from lm_agents.agents.openai_policy import OpenAIChatPolicy
from lm_agents.envs import make_env
from lm_agents.eval.runner import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None, help="only run the first N tasks")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    def env_fn():
        return make_env(cfg["env"]["name"], **cfg["env"].get("kwargs", {}))

    task_ids = cfg.get("task_ids") or env_fn().task_ids()
    task_ids = task_ids[: args.limit]
    policy = OpenAIChatPolicy(**cfg["model"])
    metrics = evaluate(
        policy, env_fn, task_ids, workers=cfg.get("workers", 1), out_dir=cfg.get("output_dir")
    )
    print(json.dumps(metrics, indent=2))
    if cfg.get("output_dir"):
        for p in make_plots([Path(cfg["output_dir"])]):
            print(f"plot: {p}")


if __name__ == "__main__":
    main()
