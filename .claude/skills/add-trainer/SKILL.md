---
name: add-trainer
description: Add or modify a post-training algorithm (SFT, OPD / on-policy distillation, RLFT such as PPO, GRPO, RLOO). Use when the user asks to implement a new training method, loss or objective.
---

# Add a trainer

1. Create `src/lm_agents/trainers/<method>.py` with a class subclassing `Trainer` (`trainers/base.py`); implement `train()`.
2. Register it in the `TRAINERS` dict in `src/lm_agents/trainers/__init__.py`. The key is the `method:` value in configs.
3. On-policy methods (OPD, RLFT) must collect data with `lm_agents.data.rollout.rollout`. Do not write a second rollout loop.
4. Multi-turn masking: compute loss only on assistant tokens; env observations are masked.
5. Read all hyperparameters from `self.cfg["train"]`. No hard-coded constants.
6. Add `configs/<method>/swebench.yaml` as the reference config.
7. Log at least: loss, mean reward (on-policy), KL to reference/teacher, mean turns per episode.
8. Add a CPU-only smoke test with a tiny model or mocked policy.

Method notes:
- **SFT**: cross-entropy on assistant turns of expert trajectories.
- **OPD**: sample from the student and minimize per-token reverse KL(student || teacher) on the sampled tokens.
- **GRPO**: group-normalized advantages over `group_size` rollouts per task; no value model.
