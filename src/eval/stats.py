"""Across-seed statistics: bootstrap CIs, Mann-Whitney U, effect sizes."""
import numpy as np
from scipy.stats import mannwhitneyu


def bootstrap_ci(x, n_boot=10000, alpha=0.05, seed=0, stat=np.mean):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    bs = np.array([stat(rng.choice(x, len(x), replace=True)) for _ in range(n_boot)])
    return float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def bootstrap_diff_ci(a, b, n_boot=10000, alpha=0.05, seed=0):
    """CI of mean(a) - mean(b), resampling each group independently."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    d = np.array([rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n_boot)])
    return float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))


def cliffs_delta(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (len(a) * len(b)))


def compare(a, b):
    """Two-sided Mann-Whitney U between seed-level values a and b."""
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    lo, hi = bootstrap_diff_ci(a, b)
    return dict(U=float(u), p=float(p), cliffs_delta=cliffs_delta(a, b), diff_mean=float(np.mean(a) - np.mean(b)),
                diff_ci_lo=lo, diff_ci_hi=hi)
