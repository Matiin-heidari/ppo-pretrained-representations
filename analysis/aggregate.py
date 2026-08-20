"""Aggregate results across all real (non-smoke) runs into report-ready tables and figures.

Reads whatever is currently under experiment_results/ (tensorboard reward curves,
dormant-ratio JSONL, generalization-eval JSON) and produces, per (task, encoder), mean +/-
95% CI across seeds:
  - a learning-curve plot (reward vs. env steps)          -> H1/H2
  - a reward-at-budget table (100k/250k/500k)              -> H2
  - a generalization-gap table (train vs. held-out test)   -> H3
  - a dormant-ratio-vs-steps plot (encoder units, tau=0.025) -> diagnostic

One set of outputs per task (e.g. oneroom, fourrooms) -- tasks are never averaged together.
Missing pieces (e.g. a generalization eval that hasn't been run on some machine yet, or an
encoder that was only run on one task) are skipped with a note printed to stdout rather than
failing the whole aggregation -- run this again after more data lands to fill the gaps in.

Usage: python -m analysis.aggregate
"""
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.stats import mean_ci95, t_crit_95

RESULTS_DIR = "experiment_results"
OUT_DIR = "analysis/aggregated"
BUDGETS = [100_000, 250_000, 500_000]
PRIMARY_TAU = "0.025"

TB_DIR_RE = re.compile(r"^(?P<run_name>.+)_\d+$")
TASK_SEED_RE = re.compile(r"^(?P<task>.+)_seed(?P<seed>\d+)$")

# Fixed categorical order (dataviz skill palette, light-mode slots 1-4) -- never reassigned
# per-plot, so an encoder keeps its color across every figure in the report, task included.
# Matched against run_name by prefix (longest-first isn't needed: names are prefix-disjoint),
# not a generic regex split, since both encoder and task segments can contain underscores.
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
TASK_LABEL = {
    "oneroom": "MiniWorld-OneRoom",
    "fourrooms": "MiniWorld-FourRooms",
}
INK = "#0b0b0b"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"


def encoder_task_seed(run_name):
    for encoder in ENCODER_ORDER:
        prefix = encoder + "_"
        if run_name.startswith(prefix):
            m = TASK_SEED_RE.match(run_name[len(prefix):])
            if m:
                return encoder, m.group("task"), int(m.group("seed"))
    return None, None, None


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


def load_reward_curve(tb_dir):
    ea = EventAccumulator(tb_dir, size_guidance={"scalars": 0})
    ea.Reload()
    events = ea.Scalars("rollout/ep_rew_mean")
    steps = np.array([e.step for e in events], dtype=float)
    values = np.array([e.value for e in events], dtype=float)
    return steps, values


def group_runs(runs):
    """task -> encoder -> seed -> (run_name, tb_dir). Tasks are kept fully separate."""
    grouped = {}
    for run_name, tb_dir in runs.items():
        encoder, task, seed = encoder_task_seed(run_name)
        if encoder is None:
            print(f"[skip] unrecognized run_name format: {run_name}")
            continue
        grouped.setdefault(task, {}).setdefault(encoder, {})[seed] = (run_name, tb_dir)
    return grouped


# -- learning curves ----------------------------------------------------------------------


def plot_learning_curves(task, by_encoder, out_path):
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    any_series = False
    for encoder in ENCODER_ORDER:
        seeds = by_encoder.get(encoder)
        if not seeds:
            continue
        curves = [load_reward_curve(tb_dir) for _run, tb_dir in seeds.values()]
        # Runs log at (near-)identical step counts (same n_steps/n_envs/total_timesteps), but
        # interpolate onto a common grid to be robust to the last iteration's step count
        # differing slightly across seeds.
        common_steps = min((s for s, _v in curves), key=len)
        interped = np.stack([np.interp(common_steps, s, v) for s, v in curves])
        mean = interped.mean(axis=0)
        ci95 = t_crit_95(len(curves)) * interped.std(axis=0, ddof=1) / np.sqrt(len(curves))

        color = ENCODER_COLOR[encoder]
        ax.plot(common_steps, mean, color=color, linewidth=2, label=ENCODER_LABEL[encoder])
        ax.fill_between(common_steps, mean - ci95, mean + ci95, color=color, alpha=0.15, linewidth=0)
        any_series = True

    if not any_series:
        print(f"[learning curves][{task}] no runs found, skipping plot")
        plt.close(fig)
        return

    ax.set_xlabel("Environment steps", color=INK)
    ax.set_ylabel("Episodic return (rollout/ep_rew_mean)", color=INK)
    ax.set_title(f"Sample efficiency on {TASK_LABEL.get(task, task)}", color=INK)
    ax.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(GRIDLINE)
    ax.tick_params(colors=INK_MUTED)
    ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[learning curves][{task}] saved {out_path}")


# -- reward at fixed budgets (H2) ----------------------------------------------------------


def budget_table(by_encoder):
    rows = []
    for encoder in ENCODER_ORDER:
        seeds = by_encoder.get(encoder)
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


def generalization_table(task, by_encoder):
    gen_dir = os.path.join(RESULTS_DIR, "generalization")
    files = glob.glob(os.path.join(gen_dir, "*.json"))
    by_encoder_gen = {}
    for path in files:
        with open(path) as f:
            data = json.load(f)
        encoder, run_task, _seed = encoder_task_seed(data["run_name"])
        if encoder is None or run_task != task:
            continue
        by_encoder_gen.setdefault(encoder, []).append(data)

    rows = []
    for encoder in ENCODER_ORDER:
        if encoder not in by_encoder:
            continue
        entries = by_encoder_gen.get(encoder)
        if not entries:
            print(f"[generalization][{task}] no eval data yet for {encoder} -- skipping")
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


def plot_dormant_ratio(task, by_encoder, out_path):
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    any_series = False
    for encoder in ENCODER_ORDER:
        seeds = by_encoder.get(encoder)
        if not seeds:
            continue
        paths = [
            os.path.join(RESULTS_DIR, "dormant", f"{run_name}.jsonl") for run_name, _tb in seeds.values()
        ]
        paths = [p for p in paths if os.path.exists(p)]
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
        ci95 = (
            t_crit_95(len(per_seed)) * interped.std(axis=0, ddof=1) / np.sqrt(len(per_seed))
            if len(per_seed) > 1
            else np.zeros_like(mean)
        )

        color = ENCODER_COLOR[encoder]
        ax.plot(common_steps, mean, color=color, linewidth=2, label=ENCODER_LABEL[encoder])
        ax.fill_between(common_steps, mean - ci95, mean + ci95, color=color, alpha=0.15, linewidth=0)
        any_series = True

    if not any_series:
        print(f"[dormant ratio][{task}] no data found, skipping plot")
        plt.close(fig)
        return

    ax.set_xlabel("Environment steps", color=INK)
    ax.set_ylabel(f"Encoder dormant ratio (tau={PRIMARY_TAU})", color=INK)
    ax.set_title(f"Dormant-neuron ratio over training on {TASK_LABEL.get(task, task)}", color=INK)
    ax.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(GRIDLINE)
    ax.tick_params(colors=INK_MUTED)
    ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[dormant ratio][{task}] saved {out_path}")


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
    grouped = group_runs(runs)

    for task, by_encoder in sorted(grouped.items()):
        print(f"=== task: {task} ===")
        for encoder in ENCODER_ORDER:
            n = len(by_encoder.get(encoder, {}))
            if n:
                print(f"  {encoder}: {n} seed(s)" + ("" if n == 3 else "  <- expected 3"))

        plot_learning_curves(task, by_encoder, os.path.join(OUT_DIR, f"learning_curves_{task}.png"))
        write_csv(budget_table(by_encoder), os.path.join(OUT_DIR, f"reward_at_budget_{task}.csv"))
        write_csv(
            generalization_table(task, by_encoder),
            os.path.join(OUT_DIR, f"generalization_gap_{task}.csv"),
        )
        plot_dormant_ratio(task, by_encoder, os.path.join(OUT_DIR, f"dormant_ratio_{task}.png"))


if __name__ == "__main__":
    main()
