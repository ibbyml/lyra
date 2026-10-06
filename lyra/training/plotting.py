"""Plot a run's loss, gradient norm, and expert routing balance: python -m lyra.training.plotting <run_dir>"""

from __future__ import annotations

import json
import sys
from itertools import accumulate

from etils.epath import Path


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _routing_series(run_dir: Path) -> tuple[list[int], list[float], list[float]]:
    """Busiest and least-used expert load relative to uniform, averaged over MoE layers."""
    steps, busiest, least_used = [], [], []
    for row in _rows(run_dir / "numerics.jsonl"):
        routers = [stats for stats in row["records"].values() if "load_max_ratio" in stats]
        if routers:
            steps.append(row["step"])
            busiest.append(sum(stats["load_max_ratio"] for stats in routers) / len(routers))
            least_used.append(sum(stats["load_min_ratio"] for stats in routers) / len(routers))
    return steps, busiest, least_used


def plot_run(run_dir: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    rows = _rows(run_dir / "metrics.jsonl")
    train = [row for row in rows if row["kind"] == "train"]
    evals = [row for row in rows if row["kind"] == "eval"]
    if not train:
        return
    tokens_per_step = train[-1]["total_tokens"] / train[-1]["step"] / 1e6
    config = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").exists() else {}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    panels = [
        ("Loss", "Cross-entropy (nats)", [(train, "ce_loss", "Train", "#2563eb"), (evals, "eval_ce_loss", "Validation", "#d97706")]),
        ("Gradient Norm (log scale)", "Global gradient L2 norm", [(train, "grad_norm", "Train", "#2563eb")]),
    ]
    for ax, (title, ylabel, lines) in zip(axes, panels):
        for rows_, key, label, color in lines:
            if not rows_:
                continue
            x = [row["step"] * tokens_per_step for row in rows_]
            y = [row[key] for row in rows_]
            if label == "Train":  # Raw values faintly, with an exponential moving average on top.
                ax.plot(x, y, color=color, alpha=0.2, linewidth=0.8)
                y = list(accumulate(y, lambda previous, value: previous + 0.2 * (value - previous)))
                ax.plot(x, y, label=label, color=color, linewidth=1.8)
            else:
                ax.plot(x, y, label=label, color=color, linewidth=1.8)
                ax.scatter(x[-1], y[-1], color=color, s=18, zorder=3)
                ax.annotate(f"{y[-1]:.2f}", (x[-1], y[-1]), xytext=(-4, 10), textcoords="offset points", ha="right", color=color)
        ax.set_title(title, fontsize=14, pad=12)
        ax.set_ylabel(ylabel, fontsize=11)
    axes[1].set_yscale("log", nonpositive="mask")

    routing = axes[2]
    steps, busiest, least_used = _routing_series(run_dir)
    if steps:
        x = [step * tokens_per_step for step in steps]
        routing.plot(x, busiest, label="Busiest", color="#7c3aed", linewidth=1.5)
        routing.plot(x, least_used, label="Least used", color="#0f766e", linewidth=1.5)
        routing.axhline(1.0, label="Uniform", color="#64748b", linestyle="--", linewidth=1)
        routing.set_ylim(bottom=0)
    else:
        routing.text(0.5, 0.5, "Routing diagnostics not recorded", transform=routing.transAxes, ha="center", color="#64748b")
    routing.set_title("Routing Balance", fontsize=14, pad=12)
    routing.set_ylabel("Expert load / uniform load", fontsize=11)

    for ax in axes:
        ax.set_xlabel("Training tokens (millions)", fontsize=11, labelpad=8)
        ax.tick_params(labelsize=10)
        ax.grid(True, color="#e2e8f0", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#cbd5e1")
        ax.margins(x=0.02)
        if ax.lines:
            ax.legend(frameon=False, fontsize=10, loc="upper right")
    fig.suptitle(config.get("model", {}).get("name", "Training progress"), fontsize=17, fontweight="bold", y=0.97)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.8, bottom=0.16, wspace=0.3)
    with (run_dir / "metrics.png").open("wb") as stream:
        fig.savefig(stream, format="png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    plot_run(sys.argv[1])
