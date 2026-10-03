"""Neural analyses on trained main_connectome runs (all seeds).

1. Activity/action correlation per neuron (ranking games, disjoint seed range).
2. Lesion: silence top-5% neurons by activity variance on ranking games vs. 20 random 5% sets,
   evaluated on the standard test games.
Outputs: results/lesion.csv, results/neuron_action_corr.csv
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd

from src.brain.build import build_spec, load_circuit
from src.eval.rollout import evaluate, trace_game
from src.train.es import TEST_BASE

RANK_BASE = 1_500_000_000
EXP = sys.argv[1] if len(sys.argv) > 1 else "main_connectome"
FRAC = 0.05
N_RANDOM = 20
N_RANK_GAMES = 20
N_TEST = 200

les_rows, corr_rows = [], []
for seed in range(10):
    run = os.path.join(ROOT, "results", "runs", EXP, f"seed{seed}")
    if not os.path.exists(os.path.join(run, "theta.npy")):
        continue
    cfg = json.load(open(os.path.join(run, "config.json")))
    theta = np.load(os.path.join(run, "theta.npy"))
    spec, _, _ = build_spec(cfg, control_seed=seed)
    _, z = load_circuit(cfg["circuit"])
    N = spec.N
    phis, acts = [], []
    for g in range(N_RANK_GAMES):
        _, a, p = trace_game(spec, theta, RANK_BASE + g, cfg["grid"], cfg["timeout_k"], record_phi=True)
        # phi recorded at step t is the state before the step-t forward pass; align with previous action
        if len(a) > 1:
            phis.append(p[1:])
            acts.append(a[:-1])
    P = np.concatenate(phis)
    Aa = np.concatenate(acts)
    var = P.var(axis=0)
    turn = np.where(Aa == 0, -1.0, np.where(Aa == 2, 1.0, 0.0))
    ps = P.std(axis=0)
    ts = turn.std()
    corr = np.where(ps > 1e-9, ((P - P.mean(0)) * (turn - turn.mean())[:, None]).mean(0) / (ps * ts + 1e-12), 0.0)
    for i in range(N):
        corr_rows.append(dict(seed=seed, neuron=i, root_id=int(z["root_ids"][i]), primary_type=str(z["primary_type"][i]),
                              super_class=str(z["super_class"][i]), activity_var=var[i], corr_turn_right=corr[i],
                              is_input=int(spec.in_group[i] >= 0), is_output=int(spec.out_group[i] >= 0)))
    k = max(1, int(round(FRAC * N)))
    seeds_test = np.arange(TEST_BASE, TEST_BASE + N_TEST)

    def score_with(mask):
        spec.mask = mask
        r = evaluate(spec, theta[None], seeds_test, cfg["grid"], cfg["timeout_k"])[0]
        return r[:, 0].mean(), r[:, 4].mean()

    base = score_with(np.ones(N))
    les_rows.append(dict(seed=seed, condition="intact", rep=0, n_silenced=0, mean_score=base[0], mean_fill=base[1]))
    top = np.argsort(-var)[:k]
    m = np.ones(N); m[top] = 0
    s = score_with(m)
    les_rows.append(dict(seed=seed, condition="top_activity_var", rep=0, n_silenced=k, mean_score=s[0], mean_fill=s[1]))
    rng = np.random.default_rng(seed + 424242)
    for rep in range(N_RANDOM):
        idx = rng.choice(N, k, replace=False)
        m = np.ones(N); m[idx] = 0
        s = score_with(m)
        les_rows.append(dict(seed=seed, condition="random", rep=rep, n_silenced=k, mean_score=s[0], mean_fill=s[1]))
    spec.mask = np.ones(N)
    print(f"seed {seed}: intact {base[0]:.2f}, top-{k} lesion {les_rows[-N_RANDOM-1]['mean_score']:.2f}, "
          f"random mean {np.mean([r['mean_score'] for r in les_rows[-N_RANDOM:]]):.2f}")

OUTD = os.path.join(ROOT, "results", "v2") if EXP.startswith("v2_") else os.path.join(ROOT, "results")
pd.DataFrame(les_rows).to_csv(os.path.join(OUTD, "lesion.csv"), index=False)
pd.DataFrame(corr_rows).to_csv(os.path.join(OUTD, "neuron_action_corr.csv"), index=False)
