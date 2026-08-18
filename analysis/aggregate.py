"""Aggregate results across all real (non-smoke) runs into report-ready tables and figures.

Reads whatever is currently under experiment_results/ (tensorboard reward curves,
dormant-ratio JSONL, generalization-eval JSON) and produces, per encoder, mean +/- 95% CI
across seeds:
  - a learning-curve plot (reward vs. env steps)          -> H1/H2
  - a reward-at-budget table (100k/250k/500k)              -> H2
  - a generalization-gap table (train vs. held-out test)   -> H3
  - a dormant-ratio-vs-steps plot (encoder units, tau=0.025) -> diagnostic

Missing pieces (e.g. a generalization eval that hasn't been run on some machine yet) are
skipped with a note printed to stdout rather than failing the whole aggregation -- run this
again after more data lands to fill the gaps in.

Usage: python -m analysis.aggregate
"""
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RESULTS_DIR = "experiment_results"
OUT_DIR = "analysis/aggregated"
BUDGETS = [100_000, 250_000, 500_000]
PRIMARY_TAU = "0.025"

RUN_NAME_RE = re.compile(r"^(?P<encoder>.+)_oneroom_seed(?P<seed>\d+)$")
TB_DIR_RE = re.compile(r"^(?P<run_name>.+)_\d+$")

# Fixed categorical order (dataviz skill palette, light-mode slots 1-4) -- never reassigned
# per-plot, so an encoder keeps its color across every figure in the report.
ENCODER_ORDER = ["cnn_scratch", "resnet18_frozen", "resnet18_finetune", "dinov2_frozen"]
ENCODER_LABEL = {
    "cnn_scratch": "CNN (scratch)",
    "resnet18_frozen": "ResNet18 (frozen)",
    "resnet18_finetune": "ResNet18 (fine-tuned)",
    "dinov2_frozen": "DINOv2 (frozen)",
}
ENCODER_COLOR = {
    "cnn_scratch": "#2a78d6",
    "resnet18_frozen": "#eb6834",
    "resnet18_finetune": "#1baf7a",
    "dinov2_frozen": "#eda100",
}
INK = "#0b0b0b"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"


def discover_runs():
    """run_name -> tensorboard event dir, for every real (non-smoke) run under experiment_results/."""
    runs = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "tensorboard", "*"))):
        base = os.path.basename(path)
        m = TB_DIR_RE.match(base)
        if not m:
            continue
        run_name = m.group("run_name")
        if run_name.startswith("_smoke"):
            continue
        if glob.glob(os.path.join(path, "events.out.tfevents.*")):
            runs[run_name] = path
    return runs


def encoder_and_seed(run_name):
    m = RUN_NAME_RE.match(run_name)
    return (m.group("encoder"), int(m.group("seed"))) if m else (None, None)


def load_reward_curve(tb_dir):
    ea = EventAccumulator(tb_dir, size_guidance={"scalars": 0})
    ea.Reload()
    events = ea.Scalars("rollout/ep_rew_mean")
    steps = np.array([e.step for e in events], dtype=float)
    values = np.array([e.value for e in events], dtype=float)
    return steps, values


def mean_ci95(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return mean, 1.96 * sem


def group_by_encoder(runs):
    grouped = {}
    for run_name, tb_dir in runs.items():
        encoder, seed = encoder_and_seed(run_name)
        if encoder is None:
            print(f"[skip] unrecognized run_name format: {run_name}")
            continue
        grouped.setdefault(encoder, {})[seed] = (run_name, tb_dir)
    return grouped


# -- learning curves ----------------------------------------------------------------------


def plot_learning_curves(grouped, out_path):
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    any_series = False
    for encoder in ENCODER_ORDER:
        seeds = grouped.get(encoder)
        if not seeds:
            continue
        curves = [load_reward_curve(tb_dir) for _run, tb_dir in seeds.values()]
        # Runs log at (near-)identical step counts (same n_steps/n_envs/total_timesteps), but
        # interpolate onto a common grid to be robust to the last iteration's step count
        # differing slightly across seeds.
        common_steps = min((s for s, _v in curves), key=len)
        interped = np.stack([np.interp(common_steps, s, v) for s, v in curves])
        mean = interped.mean(axis=0)
        ci95 = 1.96 * interped.std(axis=0, ddof=1) / np.sqrt(len(curves))

        color = ENCODER_COLOR[encoder]
        ax.plot(common_steps, mean, color=color, linewidth=2, label=ENCODER_LABEL[encoder])
        ax.fill_between(common_steps, mean - ci95, mean + ci95, color=color, alpha=0.15, linewidth=0)
        any_series = True

    if not any_series:
        print("[learning curves] no runs found, skipping plot")
        plt.close(fig)
        return

    ax.set_xlabel("Environment steps", color=INK)
    ax.set_ylabel("Episodic return (rollout/ep_rew_mean)", color=INK)
    ax.set_title("Sample efficiency: reward vs. interaction budget", color=INK)
    ax.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(GRIDLINE)
    ax.tick_params(colors=INK_MUTED)
    ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[learning curves] saved {out_path}")


# -- reward at fixed budgets (H2) ----------------------------------------------------------


def budget_table(grouped):
    rows = []
    for encoder in ENCODER_ORDER:
        seeds = grouped.get(encoder)
        if not seeds:
            continue
        curves = [load_reward_curve(tb_dir) for _run, tb_dir in seeds.values()]
        row = {"encoder": ENCODER_LABEL[encoder]}
        for budget in BUDGETS:
            vals = [v[np.argmin(np.abs(s - budget))] for s, v in curves]
            mean, ci95 = mean_ci95(vals)
            row[f"reward_at_{budget}_mean"] = mean
            row[f"reward_at_{budget}_ci95"] = ci95
            row[f"reward_at_{budget}_n"] = len(vals)
        rows.append(row)
    return rows


# -- generalization gap (H3) ---------------------------------------------------------------


def generalization_table(grouped):
    gen_dir = os.path.join(RESULTS_DIR, "generalization")
    files = glob.glob(os.path.join(gen_dir, "*.json"))
    by_encoder = {}
    for path in files:
        with open(path) as f:
            data = json.load(f)
        encoder, _seed = encoder_and_seed(data["run_name"])
        if encoder is None:
            continue
        by_encoder.setdefault(encoder, []).append(data)

    rows = []
    for encoder in ENCODER_ORDER:
        entries = by_encoder.get(encoder)
        if not entries:
            print(f"[generalization] no eval data yet for {encoder} -- skipping")
            continue
        train_mean, train_ci = mean_ci95([e["train"]["mean"] for e in entries])
        test_mean, test_ci = mean_ci95([e["test"]["mean"] for e in entries])
        gap_mean, gap_ci = mean_ci95([e["generalization_gap"] for e in entries])
        rows.append(
            {
                "encoder": ENCODER_LABEL[encoder],
                "train_mean": train_mean,
                "train_ci95": train_ci,
                "test_mean": test_mean,
                "test_ci95": test_ci,
                "gap_mean": gap_mean,
                "gap_ci95": gap_ci,
                "n_seeds": len(entries),
            }
        )
    return rows


# -- dormant ratio (diagnostic) -------------------------------------------------------------


def plot_dormant_ratio(out_path):
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    any_series = False
    for encoder in ENCODER_ORDER:
        paths = sorted(glob.glob(os.path.join(RESULTS_DIR, "dormant", f"{encoder}_oneroom_seed*.jsonl")))
        if not paths:
            continue
        per_seed = []
        for path in paths:
            steps, ratios = [], []
            with open(path) as f:
                for line in f:
                    rec = json.loads(line)
                    steps.append(rec["num_timesteps"])
                    ratios.append(rec["aggregate"]["encoder"][PRIMARY_TAU])
            per_seed.append((np.array(steps, dtype=float), np.array(ratios, dtype=float)))

        common_steps = min((s for s, _r in per_seed), key=len)
        interped = np.stack([np.interp(common_steps, s, r) for s, r in per_seed])
        mean = interped.mean(axis=0)
        ci95 = 1.96 * interped.std(axis=0, ddof=1) / np.sqrt(len(per_seed)) if len(per_seed) > 1 else np.zeros_like(mean)

        color = ENCODER_COLOR[encoder]
        ax.plot(common_steps, mean, color=color, linewidth=2, label=ENCODER_LABEL[encoder])
        ax.fill_between(common_steps, mean - ci95, mean + ci95, color=color, alpha=0.15, linewidth=0)
        any_series = True

    if not any_series:
        print("[dormant ratio] no data found, skipping plot")
        plt.close(fig)
        return

    ax.set_xlabel("Environment steps", color=INK)
    ax.set_ylabel(f"Encoder dormant ratio (tau={PRIMARY_TAU})", color=INK)
    ax.set_title("Dormant-neuron ratio over training", color=INK)
    ax.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(GRIDLINE)
    ax.tick_params(colors=INK_MUTED)
    ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[dormant ratio] saved {out_path}")


# -- CSV writer (no pandas dependency) ------------------------------------------------------


def write_csv(rows, path):
    if not rows:
        print(f"[csv] no rows for {path}, skipping")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w") as f:
        f.write(",".join(fieldnames) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in fieldnames) + "\n")
    print(f"[csv] saved {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    runs = discover_runs()
    print(f"Discovered {len(runs)} real runs: {sorted(runs)}")
    grouped = group_by_encoder(runs)
    for encoder in ENCODER_ORDER:
        n = len(grouped.get(encoder, {}))
        print(f"  {encoder}: {n} seed(s)" + ("" if n == 3 else "  <- expected 3"))

    plot_learning_curves(grouped, os.path.join(OUT_DIR, "learning_curves.png"))
    write_csv(budget_table(grouped), os.path.join(OUT_DIR, "reward_at_budget.csv"))
    write_csv(generalization_table(grouped), os.path.join(OUT_DIR, "generalization_gap.csv"))
    plot_dormant_ratio(os.path.join(OUT_DIR, "dormant_ratio.png"))


if __name__ == "__main__":
    main()
