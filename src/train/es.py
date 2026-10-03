"""CMA-ES training loop (gradient-free) with train / validation / test seed separation.

Seed ranges (disjoint):  train [0, 1e8)   validation [5e8, 5e8+1e4)   test [1e9, 1e9+1e4)
"""
import time

import cma
import numpy as np

from src.eval.rollout import evaluate

TRAIN_MAX = 100_000_000
VAL_BASE = 500_000_000
TEST_BASE = 1_000_000_000


def fitness_from_stats(res, w):
    """res: (P, S, N_STATS). Returns per-candidate fitness (P,)."""
    food = res[:, :, 0]
    fill = res[:, :, 4]
    win = res[:, :, 5]
    timeout = (res[:, :, 3] == 4).astype(float)
    f = w["food"] * food + w["fill"] * fill + w["win"] * win - w["timeout"] * timeout
    return f.mean(axis=1)


def train(spec, D, cfg, run_seed, log_path=None, init_theta=None):
    G = cfg["grid"]
    tk = cfg["timeout_k"]
    rng = np.random.default_rng(run_seed)
    x0 = np.zeros(D) if init_theta is None else init_theta
    opts = {"seed": int(run_seed) + 1, "popsize": cfg["popsize"], "verbose": -9,
            "CMA_diagonal": cfg.get("cma_diagonal", False), "maxiter": cfg["generations"]}
    es = cma.CMAEvolutionStrategy(x0, cfg["sigma0"], opts)
    val_seeds = np.arange(VAL_BASE, VAL_BASE + cfg["val_games"])
    best_val, best_theta = -np.inf, np.array(x0, float)
    rows = []
    t0 = time.time()
    total_steps = 0
    max_steps = cfg.get("max_train_steps", 0)
    for gen in range(cfg["generations"]):
        X = np.array(es.ask())
        seeds = rng.integers(0, TRAIN_MAX, cfg["games_per_eval"])  # common random numbers per gen
        res = evaluate(spec, X, seeds, G, tk)
        fit = fitness_from_stats(res, cfg["fitness"])
        es.tell(list(X), list(-fit))
        row = dict(gen=gen, time_s=time.time() - t0, fit_best=fit.max(), fit_mean=fit.mean(),
                   food_mean_pop=res[:, :, 0].mean(), steps=res[:, :, 2].sum(), sigma=es.sigma)
        if gen % cfg["val_every"] == 0 or gen == cfg["generations"] - 1:
            cand = np.stack([es.mean, X[np.argmax(fit)]])
            vr = evaluate(spec, cand, val_seeds, G, tk)
            vf = fitness_from_stats(vr, cfg["fitness"])
            j = int(np.argmax(vf))
            row.update(val_fit=vf[j], val_food=vr[j, :, 0].mean(), val_win=vr[j, :, 5].mean())
            if vf[j] > best_val:
                best_val, best_theta = vf[j], cand[j].copy()
        total_steps += row["steps"]
        stop = max_steps and total_steps >= max_steps
        row["total_steps"] = total_steps
        if stop and "val_fit" not in row:
            cand = np.stack([es.mean, X[np.argmax(fit)]])
            vr = evaluate(spec, cand, val_seeds, G, tk)
            vf = fitness_from_stats(vr, cfg["fitness"])
            j = int(np.argmax(vf))
            row.update(val_fit=vf[j], val_food=vr[j, :, 0].mean(), val_win=vr[j, :, 5].mean())
            if vf[j] > best_val:
                best_val, best_theta = vf[j], cand[j].copy()
        rows.append(row)
        if log_path is not None and (gen % 10 == 0 or gen == cfg["generations"] - 1 or stop):
            import pandas as pd
            pd.DataFrame(rows).to_csv(log_path, index=False)
        if stop:
            break
    return best_theta, rows
