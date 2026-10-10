# LM_Agents

A concise, open repo for **post-training LLM agents** on multi-turn and SWE tasks.

## Methods
- **SFT** — supervised fine-tuning on agent trajectories
- **OPD** — on-policy distillation from a stronger teacher
- **RLFT** — RL fine-tuning (PPO / GRPO) with environment rewards

## Benchmarks
| Type       | Environments                                  |
|------------|-----------------------------------------------|
| SWE        | SWE-bench (Lite / Verified)                   |
| Multi-turn | WebArena, WebShop, ScienceWorld, Search-QA    |

## Layout
```
src/lm_agents/   agents, envs, trainers (sft / opd / rlft), eval
configs/         one YAML per experiment
scripts/         launch training & evaluation
```

## Quickstart
```bash
uv venv lm_ag_venv --python 3.12 && source lm_ag_venv/bin/activate
uv pip install -e ".[dev]"
python scripts/train.py --config configs/sft/swebench.yaml
python scripts/eval.py  --config configs/eval/swebench.yaml
```

## License
MIT
