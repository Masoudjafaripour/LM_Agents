---
description: Run evaluation for a config and summarize metrics (usage: /eval <config>)
argument-hint: <configs/eval/....yaml>
---

Run `python scripts/eval.py --config $ARGUMENTS`.

Then report in a short table: env, split, model, n, mean_reward, success_rate.
If a previous result for the same env exists under `runs/`, show the delta.
If the run fails, show the error and the most likely cause; do not retry with changed settings without asking.
