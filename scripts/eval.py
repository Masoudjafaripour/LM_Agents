"""Usage: python scripts/eval.py --config configs/eval/swebench.yaml"""

import argparse
import json

import yaml

from lm_agents.envs import make_env
from lm_agents.eval.runner import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    env = make_env(cfg["env"]["name"], **cfg["env"].get("kwargs", {}))
    policy = ...  # TODO: build policy from cfg["model"]
    metrics = evaluate(policy, env, cfg.get("task_ids", []))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
