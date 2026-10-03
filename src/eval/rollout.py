"""Batched game rollouts for all policy kinds (numba, parallel over games).

Per-game statistics vector (float64[N_STATS]):
  0 score (food eaten)   1 final length   2 steps   3 cause   4 fill at end
  5 win                  6..15  steps per fill bin (10 bins of fill ratio)
  16..25 space-preserving matches per bin (chosen action has max flood area)
  26..35 informative steps per bin (flood areas of safe actions differ)
  36..45 informative matches per bin
"""
import numpy as np
from numba import njit, prange

from src.snake import engine as E
from src.brain.policies import rate_forward, mlp_forward, argmax3
from src.baselines.heuristics import greedy_action, floodfill_action, random_action

KIND_RANDOM, KIND_GREEDY, KIND_FLOOD, KIND_RATE, KIND_MLP = 0, 1, 2, 3, 4
KIND_NAMES = {"random": 0, "greedy": 1, "floodfill": 2, "rate": 3, "mlp": 4}
N_STATS = 46
NBINS = 10


class NetSpec:
    """Container for everything a rate/MLP policy needs besides theta."""

    def __init__(self, kind, feat_idx, N=1, indptr=None, indices=None, wdata=None,
                 in_group=None, out_group=None, k_in=0, k_out=0, T=10, hidden=0, mask=None, feat_version=1):
        self.kind = kind
        self.feat_idx = np.asarray(feat_idx, dtype=np.int64)
        self.N = N
        self.indptr = indptr if indptr is not None else np.zeros(N + 1, np.int64)
        self.indices = indices if indices is not None else np.zeros(0, np.int64)
        self.wdata = wdata if wdata is not None else np.zeros(0, np.float64)
        self.in_group = in_group if in_group is not None else -np.ones(N, np.int64)
        self.out_group = out_group if out_group is not None else -np.ones(N, np.int64)
        self.k_in, self.k_out, self.T, self.hidden = k_in, k_out, T, hidden
        self.out_size = np.bincount(self.out_group[self.out_group >= 0], minlength=max(k_out, 1)).astype(np.int64)
        self.mask = mask if mask is not None else np.ones(N, np.float64)
        self.feat_version = feat_version

    def args(self):
        return (self.kind, self.feat_idx, self.indptr, self.indices, self.wdata,
                self.in_group, self.out_group, self.out_size, self.k_in, self.k_out,
                self.T, self.hidden, self.mask, self.feat_version)


@njit(cache=True)
def play_one(theta, seed, G, timeout_k, kind, feat_idx, indptr, indices, wdata,
             in_group, out_group, out_size, k_in, k_out, T, hidden, mask, feat_version,
             out, rec_actions, rec_phi):
    """Play one game. If rec_actions.shape[0] > 0, record actions (and phi if
    rec_phi has rows) up to its capacity."""
    A = G * G
    st, body, occ = E.new_game(G, seed, timeout_k)
    u = np.zeros(E.N_FEATURES_MAX)
    areas = np.zeros(3)
    stack = np.zeros(A, np.int64)
    seen = np.zeros(A, np.int8)
    segidx = np.zeros(A, np.int64)
    n_feat = feat_idx.shape[0]
    u_sel = np.zeros(n_feat)
    N = indptr.shape[0] - 1
    x = np.zeros(N)
    phi = np.zeros(N)
    recv = np.zeros(N)
    drive = np.zeros(max(k_in, 1))
    rvec = np.zeros(max(k_out, 1))
    hbuf = np.zeros(max(hidden, 1))
    logits = np.zeros(3)
    prng = E.rng_seed(seed + 7777777)
    for i in range(N_STATS_):
        out[i] = 0.0
    t = 0
    while st[E.S_DONE] == 0:
        if feat_version == 1:
            E.compute_features(st, body, occ, u, stack, seen, segidx, areas)
        else:
            E.compute_features_v2(st, body, occ, u, segidx)
            for k in range(3):
                areas[k] = -1.0
        if kind == 0:
            a, prng = random_action(prng)
        elif kind == 1:
            a = greedy_action(st, body, occ)
        elif kind == 2:
            a = floodfill_action(st, body, occ, stack, seen)
        else:
            for j in range(n_feat):
                u_sel[j] = u[feat_idx[j]]
            if kind == 3:
                rate_forward(x, phi, recv, drive, u_sel, theta, n_feat, k_in, k_out,
                             indptr, indices, wdata, in_group, out_group, out_size,
                             mask, T, logits, rvec)
            else:
                mlp_forward(theta, u_sel, n_feat, hidden, hbuf, logits)
            a = argmax3(logits)
        # space-preserving bookkeeping
        fill = st[E.S_LEN] / A
        b = int(fill * NBINS)
        if b >= NBINS:
            b = NBINS - 1
        best = -1.0
        worst = 1e18
        nsafe = 0
        for k in range(3):
            if areas[k] >= 0:
                nsafe += 1
                if areas[k] > best:
                    best = areas[k]
                if areas[k] < worst:
                    worst = areas[k]
        if nsafe > 0:
            match = 1.0 if areas[a] >= 0 and areas[a] == best else 0.0
            out[6 + b] += 1.0
            out[16 + b] += match
            if nsafe >= 2 and worst < best:
                out[26 + b] += 1.0
                out[36 + b] += match
        if t < rec_actions.shape[0]:
            rec_actions[t] = a
            if rec_phi.shape[0] > t and N > 1:
                for i in range(N):
                    rec_phi[t, i] = phi[i]
        E.step(st, body, occ, a)
        t += 1
    out[0] = st[E.S_SCORE]
    out[1] = st[E.S_LEN]
    out[2] = st[E.S_STEPS]
    out[3] = st[E.S_CAUSE]
    out[4] = st[E.S_LEN] / A
    out[5] = 1.0 if st[E.S_CAUSE] == E.WIN else 0.0
    return t


N_STATS_ = N_STATS


@njit(parallel=True, cache=True)
def play_batch(thetas, seeds, G, timeout_k, kind, feat_idx, indptr, indices, wdata,
               in_group, out_group, out_size, k_in, k_out, T, hidden, mask, feat_version):
    P = thetas.shape[0]
    S = seeds.shape[0]
    res = np.zeros((P, S, N_STATS_))
    empty_a = np.zeros(0, np.int64)
    empty_p = np.zeros((0, 0))
    for job in prange(P * S):
        p = job // S
        s = job % S
        play_one(thetas[p], seeds[s], G, timeout_k, kind, feat_idx, indptr, indices, wdata,
                 in_group, out_group, out_size, k_in, k_out, T, hidden, mask, feat_version,
                 res[p, s], empty_a, empty_p)
    return res


def evaluate(spec, thetas, seeds, G=10, timeout_k=1):
    thetas = np.atleast_2d(np.asarray(thetas, dtype=np.float64))
    return play_batch(thetas, np.asarray(seeds, dtype=np.int64), G, timeout_k, *spec.args())


def trace_game(spec, theta, seed, G=10, timeout_k=1, max_steps=20000, record_phi=False):
    out = np.zeros(N_STATS)
    acts = np.zeros(max_steps, np.int64)
    phi = np.zeros((max_steps, spec.N)) if record_phi else np.zeros((0, 0))
    n = play_one(np.asarray(theta, np.float64), seed, G, timeout_k, *spec.args(), out, acts, phi)
    n = min(n, max_steps)
    return out, acts[:n], (phi[:n] if record_phi else None)
