"""Aggregate results/runs/* into paper tables (CSV)."""
import glob
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd

from src.eval.stats import bootstrap_ci, compare
from src.snake.engine import CAUSE_NAMES

R = os.path.join(ROOT, "results")
TAG = sys.argv[1] if len(sys.argv) > 1 else ""
PREFIX = f"{TAG}_" if TAG else ""
REF = f"{PREFIX}main_connectome"
OUT = os.path.join(R, TAG) if TAG else R
os.makedirs(OUT, exist_ok=True)
METRICS = ["mean_score", "median_score", "max_length", "mean_fill", "win_rate", "steps_per_food", "train_time_s"]

runs = []
for f in sorted(glob.glob(os.path.join(R, "runs", "*", "seed*", "summary.json"))):
    exp = f.split(os.sep)[-3]
    if exp.startswith(("smoke", "budget", "trial")):
        continue
    if TAG and not exp.startswith(PREFIX):
        continue
    if not TAG and exp.startswith("v2_"):
        continue
    runs.append(json.load(open(f)))
df = pd.DataFrame(runs)
df.to_csv(os.path.join(OUT, "summary_runs.csv"), index=False)

rows = []
for exp, g in df.groupby("exp"):
    r = dict(exp=exp, n_seeds=len(g), n_params=int(g.n_params.iloc[0]))
    for m in METRICS:
        r[f"{m}_mean"] = g[m].mean()
        r[f"{m}_sd"] = g[m].std(ddof=1) if len(g) > 1 else 0.0
        if len(g) > 1:
            r[f"{m}_ci_lo"], r[f"{m}_ci_hi"] = bootstrap_ci(g[m].values)
    rows.append(r)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "summary_models.csv"), index=False)

# statistical comparisons against the connectome model (seed-level mean scores)
ref = df[df.exp == REF]
comp = []
for exp, g in df.groupby("exp"):
    if exp == REF or len(g) < 2 or len(ref) < 2:
        continue
    for m in ["mean_score", "mean_fill"]:
        c = compare(ref[m].values, g[m].values)
        comp.append(dict(reference=REF, other=exp, metric=m, **c))
pd.DataFrame(comp).to_csv(os.path.join(OUT, "comparisons.csv"), index=False)

# per-game tables: death causes and space-preserving action rate
dc, sp = [], []
for exp in df.exp.unique():
    games = pd.concat([pd.read_csv(f).assign(seed=int(f.split(os.sep)[-2][4:]))
                       for f in glob.glob(os.path.join(R, "runs", exp, "seed*", "test_games.csv"))])
    for phase, sub in [("all", games), ("fill_gt_50", games[games.fill > 0.5])]:
        cnt = sub.cause.astype(int).value_counts()
        row = dict(exp=exp, phase=phase, n_games=len(sub))
        for i, name in enumerate(CAUSE_NAMES[1:], start=1):
            row[name] = cnt.get(i, 0) / max(len(sub), 1)
        dc.append(row)
    for b in range(10):
        st = games[f"bin{b}_steps"].sum()
        ist = games[f"bin{b}_inf_steps"].sum()
        sp.append(dict(exp=exp, fill_bin_lo=b / 10, steps=st, match_rate=games[f"bin{b}_match"].sum() / st if st else np.nan,
                       informative_steps=ist,
                       informative_match_rate=games[f"bin{b}_inf_match"].sum() / ist if ist else np.nan))
pd.DataFrame(dc).to_csv(os.path.join(OUT, "death_causes.csv"), index=False)
pd.DataFrame(sp).to_csv(os.path.join(OUT, "space_preserving.csv"), index=False)
print(pd.DataFrame(rows)[["exp", "n_seeds", "mean_score_mean", "mean_score_sd", "win_rate_mean", "train_time_s_mean"]].to_string())
