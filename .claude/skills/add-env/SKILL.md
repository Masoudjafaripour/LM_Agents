---
name: add-env
description: Wrap a new benchmark (e.g. WebShop, ScienceWorld, WebArena, Search-QA) as an Env in this repo. Use when the user asks to add, integrate or support a new benchmark or environment.
---

# Add a benchmark env

1. Create `src/lm_agents/envs/<name>/env.py` with a class subclassing `Env` from `lm_agents.envs.base`:
   - `reset(task_id) -> str`: initial observation (task prompt).
   - `step(action) -> StepResult`: apply the action, return observation, reward, done, info.
   - Set `max_turns` to the benchmark's usual horizon.
   - Decorate with `@register_env("<name>")`.
2. Export the class in `src/lm_agents/envs/<name>/__init__.py`.
3. Add `from lm_agents.envs import <name>` to the import block in `src/lm_agents/envs/__init__.py`.
4. Rewards: terminal reward in `[0, 1]`; put per-step diagnostics in `info`, not `reward`, unless shaping is intended.
5. Add `configs/eval/<name>.yaml`.
6. Add a test in `tests/` using a fake policy that checks `reset`/`step` contracts without network or GPU.
7. Run `pytest -q` and `ruff check .`.

Heavy dependencies (docker, browsers, simulators) go in a `[project.optional-dependencies]` extra in `pyproject.toml`, and the import stays inside the env module.
