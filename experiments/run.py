"""Train (if needed) and test one model/seed.

usage: python experiments/run.py --exp NAME --seed S [--set key=value ...]
Outputs: results/runs/NAME/seed{S}/{config.json, train_log.csv, theta.npy, test_games.csv, summary.json}
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd
import yaml

from src.brain.build import build_spec
from src.eval.rollout import evaluate, NBINS
from src.train.es import train, TEST_BASE


def load_cfg(sets, base="base"):
    cfg = yaml.safe_load(open(os.path.join(ROOT, f"experiments/configs/{base}.yaml")))
    for s in sets:
        k, v = s.split("=", 1)
        cfg_v = yaml.safe_load(v)
        if "." in k:
            a, b = k.split(".")
            cfg[a][b] = cfg_v
        else:
            cfg[k] = cfg_v
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--base", default="base")
    a = ap.parse_args()
    cfg = load_cfg(a.set, a.base)
    out = os.path.join(ROOT, "results", "runs", a.exp, f"seed{a.seed}")
    os.makedirs(out, exist_ok=True)
    spec, D, info = build_spec(cfg, control_seed=a.seed)
    cfg["n_params"] = D
    cfg["build_info"] = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in info.items()}
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    t0 = time.time()
    if D > 0:
        theta, _ = train(spec, D, cfg, run_seed=a.seed, log_path=os.path.join(out, "train_log.csv"))
        np.save(os.path.join(out, "theta.npy"), theta)
    else:
        theta = np.zeros(1)
    train_time = time.time() - t0
    seeds = np.arange(TEST_BASE, TEST_BASE + cfg["test_games"])
    t1 = time.time()
    r = evaluate(spec, theta[None, :], seeds, cfg["grid"], cfg["timeout_k"])[0]
    test_time = time.time() - t1
    cols = ["score", "length", "steps", "cause", "fill", "win"]
    cols += [f"bin{b}_steps" for b in range(NBINS)] + [f"bin{b}_match" for b in range(NBINS)]
    cols += [f"bin{b}_inf_steps" for b in range(NBINS)] + [f"bin{b}_inf_match" for b in range(NBINS)]
    df = pd.DataFrame(r, columns=cols)
    df.insert(0, "game_seed", seeds)
    df.to_csv(os.path.join(out, "test_games.csv"), index=False)
    summ = dict(exp=a.exp, seed=a.seed, model=cfg["model"], n_params=D, train_time_s=train_time,
                test_time_s=test_time, mean_score=df.score.mean(), median_score=df.score.median(),
                max_length=int(df.length.max()), mean_fill=df.fill.mean(), win_rate=df.win.mean(),
                steps_per_food=df.steps.sum() / max(df.score.sum(), 1))
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
