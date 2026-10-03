"""Export demo/weights.json (+ weights.js for file:// use).

Uses the connectome run with the highest mean test score and the degree-preserving
shuffle run with the same training seed.
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd

from src.brain.build import build_spec
from src.eval.rollout import trace_game
from src.export.export_weights import export_model

RANK_BASE = 1_500_000_000
TAG = sys.argv[1] if len(sys.argv) > 1 else ""
P = f"{TAG}_" if TAG else ""
runs = pd.read_csv(os.path.join(ROOT, "results", TAG, "summary_runs.csv"))
# median-performing connectome seed (not the best) so the demo is representative
cr = runs[runs.exp == f"{P}main_connectome"].sort_values("mean_score").reset_index(drop=True)
best_seed = int(cr.iloc[(len(cr) - 1) // 2].seed)

models = {}
for exp, label in [(f"{P}main_connectome", "FlyWire subcircuit"),
                   (f"{P}main_degree_shuffle", "Degree-preserving shuffled graph")]:
    run = os.path.join(ROOT, "results", "runs", exp, f"seed{best_seed}")
    cfg = json.load(open(os.path.join(run, "config.json")))
    theta = np.load(os.path.join(run, "theta.npy"))
    spec, _, _ = build_spec(cfg, control_seed=best_seed)
    phis = []
    for g in range(20):
        _, a, p = trace_game(spec, theta, RANK_BASE + g, cfg["grid"], cfg["timeout_k"], record_phi=True)
        phis.append(p[1:])
    var = np.concatenate(phis).var(0)
    k = max(1, int(round(0.05 * spec.N)))
    summ = json.load(open(os.path.join(run, "summary.json")))
    m = export_model(cfg, theta, control_seed=best_seed,
                     extra=dict(label=label, seed=best_seed, lesion_top=np.argsort(-var)[:k].tolist(),
                                role_out=(spec.out_group >= 0).astype(int).tolist(),
                                test_mean_score=summ["mean_score"]))
    models[exp] = m
    print(exp, "seed", best_seed, "test mean", summ["mean_score"])

data = dict(default=f"{P}main_connectome", models=models)
os.makedirs(os.path.join(ROOT, "demo"), exist_ok=True)
js = json.dumps(data, separators=(",", ":"))
open(os.path.join(ROOT, "demo", "weights.json"), "w").write(js)
open(os.path.join(ROOT, "demo", "weights.js"), "w").write("window.FLY_WEIGHTS=" + js + ";\n")
