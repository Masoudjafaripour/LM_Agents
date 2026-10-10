---
name: analyze-run
description: Analyze a training or evaluation run under runs/ (metrics, logs, sampled trajectories) and report regressions or failure modes. Use when the user asks why a run is bad, to compare runs, or to inspect agent trajectories.
---

# Analyze a run

1. Locate the run dir under `runs/` and its config. Note method, model, env and key hyperparameters.
2. Metrics: report the trend of loss, reward, KL and episode length. Flag reward collapse, KL blow-up, or lengths pinned at `max_turns`.
3. Trajectories: sample about 10 failures and 5 successes. Bucket failures, e.g. invalid tool call, loops or repeated actions, early submit, context overflow, wrong edit.
4. If a baseline run is given, show metric deltas and which failure buckets grew or shrank.
5. Finish with 2–3 concrete next experiments, each as a config change.

Keep the report short, with one table and bullets. Quote at most a few lines of any trajectory.
