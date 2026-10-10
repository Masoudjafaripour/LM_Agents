# CLAUDE.md

Post-training LLM agents (SFT / OPD / RLFT) on multi-turn and SWE benchmarks.

## Layout
- `src/lm_agents/envs/`: one subpackage per benchmark, all implementing `Env` (`envs/base.py`) and registered with `@register_env`.
- `src/lm_agents/data/rollout.py`: the single multi-turn rollout loop. Every on-policy method uses it.
- `src/lm_agents/trainers/`: `sft.py`, `opd.py`, `rlft.py`, selected by `method:` in the config.
- `configs/<method>/<env>.yaml`: one YAML per experiment. Hyperparameters live here, not in code.

## Commands
- Env: `uv venv lm_ag_venv --python 3.12 && source lm_ag_venv/bin/activate`
- Install: `uv pip install -e ".[dev]"` (training deps: `".[train]"`)
- Test: `pytest -q`
- Lint: `ruff check . && ruff format .`
- Train: `python scripts/train.py --config <yaml>`. Ask before launching; it uses GPUs.

## Conventions
- Loss is computed on assistant tokens only; env observations are masked.
- Rewards are in `[0, 1]`; diagnostics go in `StepResult.info`.
- Never commit `data/`, `runs/` or checkpoints.
- Skills: `add-env`, `add-trainer`, `analyze-run` in `.claude/skills/`.
