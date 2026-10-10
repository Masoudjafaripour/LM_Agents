"""Plot eval results. One run dir -> plots saved inside it; several -> a comparison.

Usage:
  python scripts/plot.py runs/eval_humaneval_qwen2.5-coder-7b
  python scripts/plot.py runs/eval_a runs/eval_b --out runs/plots
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Fixed categorical order (dataviz reference palette); a run keeps its slot by argv position.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
UNSOLVED = "#b5b4ae"  # neutral gray, not a series color
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})  # fmt: skip


def load_run(run_dir: Path) -> tuple[dict, list[dict]]:
    metrics = json.loads((run_dir / "metrics.json").read_text())
    with open(run_dir / "trajectories.jsonl") as f:
        trajs = [json.loads(line) for line in f]
    return metrics, trajs


def solved_by_turn(metrics: dict) -> list[float]:
    k = 1
    out = []
    while f"solved_by_turn_{k}" in metrics:
        out.append(100 * metrics[f"solved_by_turn_{k}"])
        k += 1
    return out


def plot_solved_by_turn(runs: dict[str, dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (name, metrics) in enumerate(runs.items()):
        ys = solved_by_turn(metrics)
        xs = range(1, len(ys) + 1)
        color = SERIES[i % len(SERIES)]
        ax.plot(xs, ys, color=color, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, label=name)
        if len(runs) == 1:  # few points: label each; with several runs, the legend + last value
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.1f}%", (x, y), xytext=(0, 10), textcoords="offset points",
                            ha="center", color=INK)  # fmt: skip
        else:
            ax.annotate(f"{ys[-1]:.1f}%", (xs[-1], ys[-1]), xytext=(8, 0),
                        textcoords="offset points", va="center", color=INK)  # fmt: skip
    ax.set_xticks(range(1, max(len(solved_by_turn(m)) for m in runs.values()) + 1))
    ax.set_ylim(0, 105)  # full scale, so small multi-turn gains aren't exaggerated
    ax.set_xlabel("Turns allowed")
    ax.set_ylabel("Tasks solved (%)")
    ax.grid(axis="x", visible=False)
    title = next(iter(runs)) if len(runs) == 1 else "Solved within k turns"
    ax.set_title(title)
    if len(runs) > 1:
        ax.legend(frameon=False, loc="lower right")
    ax.margins(x=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_outcomes(runs: dict[str, list[dict]], path: Path) -> None:
    """Stacked horizontal bar per run: solved at turn 1, 2, ..., unsolved."""
    counts = {name: Counter(t["solved_at"] for t in trajs) for name, trajs in runs.items()}
    max_turn = max((k for c in counts.values() for k in c if k is not None), default=1)
    cats = [(k, f"Turn {k}", SERIES[k - 1]) for k in range(1, max_turn + 1)]
    cats.append((None, "Unsolved", UNSOLVED))

    fig, ax = plt.subplots(figsize=(7, 1.6 + 0.6 * len(runs)))
    names = list(runs)
    for y, name in enumerate(names):
        n, left = len(runs[name]), 0.0
        for key, label, color in cats:
            w = 100 * counts[name].get(key, 0) / n
            if w == 0:
                continue
            ax.barh(y, w, left=left, color=color, height=0.6, edgecolor=SURFACE, linewidth=2)
            if w >= 6:
                ax.text(left + w / 2, y, f"{w:.0f}%", ha="center", va="center", color=SURFACE,
                        fontsize=9, fontweight="bold")  # fmt: skip
            left += w
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.grid(axis="y", visible=False)
    ax.set_title("Turn at which each task was first solved")
    handles = [Patch(color=color, label=label) for _, label, color in cats]
    fig.legend(handles=handles, frameon=False, ncol=len(cats), loc="lower center", handlelength=1)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_plots(run_dirs: list[Path], out: Path | None = None) -> list[Path]:
    out = out or run_dirs[0]
    out.mkdir(parents=True, exist_ok=True)
    loaded = {d.name.removeprefix("eval_"): load_run(d) for d in run_dirs}
    paths = [out / "solved_by_turn.png", out / "outcomes.png"]
    plot_solved_by_turn({k: m for k, (m, _) in loaded.items()}, paths[0])
    plot_outcomes({k: t for k, (_, t) in loaded.items()}, paths[1])
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="default: first run dir")
    args = parser.parse_args()
    for p in make_plots(args.run_dirs, args.out):
        print(p)


if __name__ == "__main__":
    main()
