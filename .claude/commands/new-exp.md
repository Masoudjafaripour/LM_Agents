---
description: Create a new experiment config (usage: /new-exp <method> <env> [notes])
argument-hint: <sft|opd|rlft> <env> [notes]
---

Create a new experiment config for: $ARGUMENTS

1. Copy the closest existing file in `configs/<method>/` as a template.
2. Name it `configs/<method>/<env>_<short-tag>.yaml`; set `output_dir: runs/<method>_<env>_<short-tag>`.
3. Make sure `env.name` is registered in `ENV_REGISTRY` (`src/lm_agents/envs/__init__.py`).
4. Show the diff vs. the template and the exact `python scripts/train.py --config ...` command. Do not launch training.
