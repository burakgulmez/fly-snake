"""Build NetSpec objects for the connectome model and its controls."""
import os

import numpy as np
import scipy.sparse as sp

from src.brain.policies import rate_param_count, mlp_hidden_for_budget, mlp_param_count
from src.connectome.subcircuit import make_groups, spectral_radius
from src.eval.rollout import NetSpec, KIND_RATE, KIND_MLP, KIND_NAMES
from src.snake import engine as E

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

FEATURE_SETS = {
    "food_danger": np.concatenate([E.FEAT_FOOD, E.FEAT_DANGER]),
    "food_danger_space": np.concatenate([E.FEAT_FOOD, E.FEAT_DANGER, E.FEAT_SPACE]),
    "full": np.arange(E.N_FEATURES),
    # version-2 sensors (food vector + obstacle rays, no look-ahead search)
    "v2_full": np.arange(E.N_FEATURES_V2),
    "v2_no_rays": np.arange(0, 10),  # food (7) + immediate danger (3)
}
FEATURE_VERSION = {"food_danger": 1, "food_danger_space": 1, "full": 1, "v2_full": 2, "v2_no_rays": 2}


def load_circuit(name):
    z = np.load(os.path.join(ROOT, "data", "processed", f"{name}.npz"))
    n = int(z["n"])
    W = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=(n, n))
    return W, z


def _degree_preserving_shuffle(W, rng, n_swaps_per_edge=10):
    """Maslov-Sneppen double-edge swaps on the directed graph; each edge keeps its
    weight (hence sign), so every neuron keeps in/out degree and its outgoing sign
    composition is not preserved exactly but global E/I edge fraction is."""
    C = W.tocoo()
    pre, post, w = C.col.copy(), C.row.copy(), C.data.copy()
    m = len(w)
    edges = set(zip(pre.tolist(), post.tolist()))
    n_try = n_swaps_per_edge * m
    a_idx = rng.integers(0, m, n_try)
    b_idx = rng.integers(0, m, n_try)
    done = 0
    for a, b in zip(a_idx, b_idx):
        if a == b:
            continue
        p1, q1, p2, q2 = pre[a], post[a], pre[b], post[b]
        if p1 == q2 or p2 == q1 or (p1, q2) in edges or (p2, q1) in edges:
            continue
        edges.discard((p1, q1)); edges.discard((p2, q2))
        edges.add((p1, q2)); edges.add((p2, q1))
        post[a], post[b] = q2, q1
        done += 1
    n = W.shape[0]
    return sp.csr_matrix((w, (post, pre)), shape=(n, n)), done


def _erdos_renyi(W, rng):
    """Same n and edge count, uniformly random directed edges (no self loops); the
    multiset of connectome weights (magnitudes and signs) is re-assigned at random."""
    n = W.shape[0]
    m = W.nnz
    chosen = set()
    pre = np.empty(m, np.int64); post = np.empty(m, np.int64)
    k = 0
    while k < m:
        p, q = rng.integers(0, n, 2)
        if p == q or (p, q) in chosen:
            continue
        chosen.add((p, q)); pre[k] = p; post[k] = q; k += 1
    w = rng.permutation(W.tocoo().data)
    return sp.csr_matrix((w, (post, pre)), shape=(n, n))


def _sign_shuffle(W, rng):
    W = W.tocsr().copy()
    W.data = np.abs(W.data) * rng.permutation(np.sign(W.data))
    return W


def build_spec(cfg, control_seed=0):
    """cfg keys: model, circuit, feature_set, rho, k_in, k_out, T."""
    feat = FEATURE_SETS[cfg["feature_set"]]
    F = len(feat)
    model = cfg["model"]
    info = {}
    if model in ("random", "greedy", "floodfill"):
        return NetSpec(KIND_NAMES[model], feat, feat_version=FEATURE_VERSION[cfg["feature_set"]]), 0, info
    W, z = load_circuit(cfg["circuit"])
    k_in, k_out = cfg["k_in"], cfg["k_out"]
    D = rate_param_count(F, k_in, k_out)
    if model == "mlp":
        h = mlp_hidden_for_budget(F, D)
        info["hidden"] = h
        return NetSpec(KIND_MLP, feat, hidden=h, feat_version=FEATURE_VERSION[cfg["feature_set"]]), mlp_param_count(F, h), info
    rng = np.random.default_rng(10_000 + control_seed)
    ig = make_groups(z["side"], z["primary_type"], z["root_ids"], z["inputs"], k_in)
    og = make_groups(z["side"], z["primary_type"], z["root_ids"], z["outputs"], k_out)
    if model == "connectome":
        pass
    elif model == "degree_shuffle":
        W, info["swaps"] = _degree_preserving_shuffle(W, rng)
    elif model == "erdos_renyi":
        W = _erdos_renyi(W, rng)
    elif model == "sign_shuffle":
        W = _sign_shuffle(W, rng)
    elif model == "no_recurrence":
        W = sp.csr_matrix(W.shape)
        og = ig.copy()  # read out directly from the driven input neurons
    else:
        raise ValueError(model)
    W = W.tocsr()
    W.sum_duplicates()
    W.sort_indices()
    rho0 = spectral_radius(W)
    if rho0 > 0:
        W = W * (cfg["rho"] / rho0)
    W = W.tocsr()
    info.update(rho_raw=rho0, edges=int(W.nnz), frac_exc=float((W.data > 0).mean()) if W.nnz else 0.0)
    spec = NetSpec(KIND_RATE, feat, W.shape[0], W.indptr.astype(np.int64), W.indices.astype(np.int64),
                   W.data.astype(np.float64), ig, og, k_in, k_out, cfg["T"],
                   feat_version=FEATURE_VERSION[cfg["feature_set"]])
    return spec, D, info
