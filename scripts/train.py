"""Usage: python scripts/train.py --config configs/sft/swebench.yaml"""

import argparse

import yaml

from lm_agents.trainers import TRAINERS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    TRAINERS[cfg["method"]](cfg).train()


if __name__ == "__main__":
    main()
