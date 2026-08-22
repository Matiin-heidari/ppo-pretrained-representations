"""Report figures and LaTeX tables for one task.

Kept separate from analysis/aggregate.py: that script is the exploratory cross-run view,
this one produces the exact assets the report includes, sized and styled for the page.

Figures are written as PDF (vector, what LaTeX should include) and PNG (quick inspection).
Tables are emitted as LaTeX table bodies so numbers in the report are regenerated rather
than transcribed by hand.

Usage: python -m analysis.plots_report [oneroom|fourrooms]
"""
import os
import sys
import json
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.aggregate import (
    ENCODER_ORDER, ENCODER_LABEL, ENCODER_COLOR, TASK_LABEL,
    RESULTS_DIR, PRIMARY_TAU, discover_runs, group_runs, load_reward_curve,
    INK, INK_MUTED, GRIDLINE, SURFACE,
)
from src.stats import mean_ci95, t_crit_95

OUT_DIR = "documents/figures"

Y_FLOOR = 5e-4  # log-axis floor for the dormancy plot

# Return thresholds for the "steps to reach" columns, per task. OneRoom saturates near 0.95
# while FourRooms tops out around 0.5, so one shared set is uninformative on one of them.
THRESHOLDS = {"oneroom": (0.5, 0.8, 0.9), "fourrooms": (0.2, 0.3, 0.4)}

# Upper limit of the return axis, per task. OneRoom uses the full range because it reaches it;
# FourRooms tops out near 0.5, where a 0--1 axis would leave half the panel empty.
Y_MAX = {"oneroom": 1.05, "fourrooms": 0.6}

# A threshold counts as reached only once the curve holds it for this many further steps.
# FourRooms returns oscillate by ~0.1, so a bare first crossing records a transient spike as
# an attainment -- the CNN touches 0.4 once near 85k and then sits at 0.29 for the rest of
# training. Requiring persistence also keeps the number readable off the figure.
HOLD_STEPS = 50_000

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.6,
})


def _seed_curves(by_encoder, encoder, tag="rollout/ep_rew_mean"):
    seeds = by_encoder.get(encoder)
    if not seeds:
        return None
    if tag == "rollout/ep_rew_mean":
        return [load_reward_curve(tb) for _run, tb in seeds.values()]
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    out = []
    for _run, tb in seeds.values():
        ea = EventAccumulator(tb, size_guidance={"scalars": 0}); ea.Reload()
        e = ea.Scalars(tag)
        out.append((np.array([x.step for x in e], float), np.array([x.value for x in e], float)))
    return out


def _band(curves):
    """Common step grid, cross-seed mean, and t-based 95% half-width at each step."""
    grid = min((s for s, _v in curves), key=len)
    stacked = np.stack([np.interp(grid, s, v) for s, v in curves])
    mean = stacked.mean(axis=0)
    n = len(curves)
    half = t_crit_95(n) * stacked.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    return grid, mean, half


def _style_axes(ax):
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRIDLINE)
    ax.tick_params(colors=INK_MUTED, length=3)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK)


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        path = os.path.join(OUT_DIR, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight", facecolor=SURFACE, **kw)
    plt.close(fig)
    print(f"  saved {OUT_DIR}/{name}.pdf (+ .png)")


def fig_learning_curves(task, by_encoder, zoom_to=100_000, show_band=False):
    """Two panels: the full budget, and the window where learning actually happens.

    show_band draws the cross-seed 95% interval. Off by default: with four series the filled
    regions overlap into an unreadable wash, and the same intervals are reported exactly in the
    summary table, which is the better place to read a number off anyway.
    """
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(5.5, 2.3), facecolor=SURFACE)
    for ax in (ax_full, ax_zoom):
        ax.set_facecolor(SURFACE)
    for encoder in ENCODER_ORDER:
        curves = _seed_curves(by_encoder, encoder)
        if not curves:
            continue
        grid, mean, half = _band(curves)
        for ax, lim in ((ax_full, grid[-1]), (ax_zoom, zoom_to)):
            m = grid <= lim
            ax.plot(grid[m] / 1000, mean[m], color=ENCODER_COLOR[encoder],
                    label=ENCODER_LABEL[encoder])
            if show_band:
                ax.fill_between(grid[m] / 1000, (mean - half)[m], (mean + half)[m],
                                color=ENCODER_COLOR[encoder], alpha=0.15, linewidth=0)
    for ax, title in ((ax_full, "Full budget"), (ax_zoom, f"First {zoom_to//1000}k steps")):
        # Return is bounded above by 1.0 (reach the goal instantly); never draw past that.
        ax.set_ylim(-0.02, Y_MAX[task])
        ax.set_xlabel("Environment steps (thousands)")
        ax.set_title(title, color=INK)
        _style_axes(ax)
    ax_full.set_ylabel("Episodic return")
    # "best" picks the least-overlapping corner from the data itself; a hand-tuned corner
    # goes wrong as soon as the axis limits or the set of encoders changes.
    ax_zoom.legend(frameon=False, labelcolor=INK, loc="best")
    fig.tight_layout(pad=0.4)
    _save(fig, f"learning_curves_{task}")


def fig_dormant(task, by_encoder):
    """Encoder dormant ratio over training. Log y: the values span three orders of magnitude."""
    fig, ax = plt.subplots(figsize=(3.4, 2.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for encoder in ENCODER_ORDER:
        paths = sorted(glob.glob(os.path.join(RESULTS_DIR, "dormant", f"{encoder}_{task}_seed*.jsonl")))
        if not paths:
            continue
        per_seed = []
        for p in paths:
            steps, ratios = [], []
            for line in open(p):
                rec = json.loads(line)
                steps.append(rec["num_timesteps"])
                ratios.append(rec["aggregate"]["encoder"][PRIMARY_TAU])
            per_seed.append((np.array(steps, float), np.array(ratios, float)))
        grid, mean, half = _band(per_seed)
        ax.plot(grid / 1000, mean, color=ENCODER_COLOR[encoder], label=ENCODER_LABEL[encoder])
        ax.fill_between(grid / 1000, np.maximum(mean - half, Y_FLOOR), mean + half,
                        color=ENCODER_COLOR[encoder], alpha=0.15, linewidth=0)
    ax.set_yscale("log")
    # Frozen DINOv2 at ~9e-4 is the smallest value present; without an explicit floor a
    # clamped CI band stretches the axis over two empty decades.
    ax.set_ylim(Y_FLOOR, 1.0)
    ax.set_xlabel("Environment steps (thousands)")
    ax.set_ylabel(rf"Encoder dormant ratio ($\tau$={PRIMARY_TAU})")
    _style_axes(ax)
    ax.legend(frameon=False, labelcolor=INK, loc="upper right")
    fig.tight_layout(pad=0.4)
    _save(fig, f"dormant_ratio_{task}")


def _fmt_steps(x):
    """Thousands of steps, or a dash when at least one seed never reached the threshold."""
    return "---" if np.isnan(x) else f"{x/1000:.1f}k"


def _steps_to(curves, threshold, hold=HOLD_STEPS):
    """First step at which the cross-seed mean curve reaches `threshold` and holds it for
    `hold` further steps. Computed on the same mean curve the figures plot, so every entry in
    the table can be checked against them. nan if that never happens."""
    grid, mean, _half = _band(curves)
    for i, step in enumerate(grid):
        if mean[i] < threshold:
            continue
        window = mean[(grid >= step) & (grid <= step + hold)]
        if len(window) and (window >= threshold).all():
            return float(step)
    return float("nan")


def table_summary(task, by_encoder):
    """Sample efficiency, final return, and PPO's trust-region diagnostics."""
    rows = []
    for encoder in ENCODER_ORDER:
        curves = _seed_curves(by_encoder, encoder)
        if not curves:
            continue
        fm, fc = mean_ci95([v[-10:].mean() for _s, v in curves])
        kl = np.mean([v[len(v) // 2:].mean() for _s, v in _seed_curves(by_encoder, encoder, "train/approx_kl")])
        cf = np.mean([v[len(v) // 2:].mean() for _s, v in _seed_curves(by_encoder, encoder, "train/clip_fraction")])
        th = THRESHOLDS[task]
        rows.append((ENCODER_LABEL[encoder], _steps_to(curves, th[0]), _steps_to(curves, th[1]),
                     _steps_to(curves, th[2]), fm, fc, kl, cf))
    lines = [r"\begin{tabular}{lrrrcrr}", r"\toprule",
             r"Encoder & \multicolumn{3}{c}{Steps to return} & Final return & "
             r"$\mathrm{KL}$ & Clip frac. \\",
             r"\cmidrule(lr){2-4}",
             f" & {th[0]} & {th[1]} & {th[2]} & (mean $\\pm$ CI) & & " + r"\\", r"\midrule"]
    for n, t5, t8, t9, fm, fc, kl, cf in rows:
        lines.append(f"{n} & {_fmt_steps(t5)} & {_fmt_steps(t8)} & {_fmt_steps(t9)} & "
                     f"${fm:.3f} \\pm {fc:.3f}$ & {kl:.3f} & {cf:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_generalization(task):
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Encoder & Train & Test & Gap (train $-$ test) \\", r"\midrule"]
    any_row = False
    for encoder in ENCODER_ORDER:
        files = sorted(glob.glob(os.path.join(RESULTS_DIR, "generalization", f"{encoder}_{task}_seed*_500000.json")))
        if not files:
            continue
        data = [json.load(open(f)) for f in files]
        tm, tc = mean_ci95([d["train"]["mean"] for d in data])
        em, ec = mean_ci95([d["test"]["mean"] for d in data])
        gm, gc = mean_ci95([d["generalization_gap"] for d in data])
        lines.append(f"{ENCODER_LABEL[encoder]} & ${tm:.3f} \\pm {tc:.3f}$ & "
                     f"${em:.3f} \\pm {ec:.3f}$ & ${gm:+.3f} \\pm {gc:.3f}$ \\\\")
        any_row = True
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) if any_row else None


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "oneroom"
    grouped = group_runs(discover_runs())
    if task not in grouped:
        raise SystemExit(f"no runs for task {task!r}; have {sorted(grouped)}")
    by_encoder = grouped[task]
    print(f"{TASK_LABEL.get(task, task)}: " +
          ", ".join(f"{e} ({len(by_encoder[e])} seeds)" for e in ENCODER_ORDER if e in by_encoder))

    fig_learning_curves(task, by_encoder)
    fig_dormant(task, by_encoder)

    os.makedirs(OUT_DIR, exist_ok=True)
    for name, body in (("summary", table_summary(task, by_encoder)),
                       ("generalization", table_generalization(task))):
        if body is None:
            print(f"  [skip] no {name} data for {task}")
            continue
        path = os.path.join(OUT_DIR, f"table_{name}_{task}.tex")
        open(path, "w").write(body + "\n")
        print(f"  saved {path}")
        print(body)


if __name__ == "__main__":
    main()
