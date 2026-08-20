"""Small-sample confidence intervals for cross-seed and cross-episode aggregation.

Split out because `analysis/aggregate.py` and `src/eval_generalization.py` both need it and
must not drift apart. Deliberately numpy-only: the analysis layer imports this, and pulling in
torch/SB3 just to compute a CI would make plotting depend on the training stack.
"""
import numpy as np

# t(0.975, df) -- two-sided 95%. Tabulated rather than computed because scipy is not a
# dependency of this project and the inverse-t needs an incomplete beta to evaluate.
# df=1 and df=2 are exact (Cauchy / closed form); the rest are the standard table.
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
    9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
    23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045,
    30: 2.042, 40: 2.021, 50: 2.009, 60: 2.000, 80: 1.990, 100: 1.984, 120: 1.980,
}
_Z95 = 1.96  # df -> inf


def t_crit_95(n) -> float:
    """Two-sided 95% critical value for a mean of `n` samples (df = n-1).

    Only three values are actually reached today: df=2 (the 3-seed grid), df=29 and df=99 (the
    30- and 100-episode generalization evals). The table covers the range anyway, because the
    bug this replaced was exactly a constant that happened to suit one sample size and was
    silently wrong for another -- hardcoding today's three would recreate it.

    At n=3 the normal 1.96 understates the interval by 2.2x: +/-1.96*SEM covers 81% at df=2,
    not 95%. Values above df=30 are interpolated in 1/df, where t is very nearly linear.
    """
    n = int(n)
    if n < 2:
        return 0.0
    df = n - 1
    if df in _T95:  # exact for df <= 30 and at each anchor above it
        return _T95[df]
    lo = max(d for d in _T95 if d <= df)
    hi = min((d for d in _T95 if d > df), default=None)
    if hi is None:  # past the last anchor, decay toward the normal limit
        return _Z95 + (_T95[lo] - _Z95) * (120.0 / df)
    w = (1.0 / df - 1.0 / lo) / (1.0 / hi - 1.0 / lo)
    return _T95[lo] + w * (_T95[hi] - _T95[lo])


def mean_ci95(values):
    """(mean, half-width of the two-sided 95% CI) over `values`."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return mean, t_crit_95(n) * sem
