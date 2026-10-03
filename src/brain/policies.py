"""Policies: connectome rate network, MLP, and heuristic wrappers (numba).

Rate model (Euler, per game step T inner steps, state kept across game steps):
    x <- x + alpha * ( -x + g * W phi(x) + I_in )
    phi = tanh,   alpha = dt/tau in (0,1) (trained),  g in (0,2) (trained)
    I_in[i] = W_in[group(i)] . u + b_in[group(i)]   for input neurons, else 0
    logits = W_out . r + b_out,  r_k = mean_{i in out-group k} phi(x_i)
W is fixed (connectome-derived, spectral radius normalised); only the interface
(W_in, b_in, W_out, b_out, alpha, g) is trained.
"""
import numpy as np
from numba import njit


def rate_param_count(n_feat, k_in, k_out):
    return k_in * n_feat + k_in + 3 * k_out + 3 + 2


def mlp_param_count(n_feat, hidden):
    return (n_feat + 1) * hidden + (hidden + 1) * 3


def mlp_hidden_for_budget(n_feat, budget):
    """Largest hidden size whose parameter count does not exceed budget."""
    h = 1
    while mlp_param_count(n_feat, h + 1) <= budget:
        h += 1
    return h


@njit(cache=True)
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


@njit(cache=True)
def rate_unpack(theta, n_feat, k_in, k_out):
    p = 0
    W_in = theta[p:p + k_in * n_feat].reshape(k_in, n_feat)
    p += k_in * n_feat
    b_in = theta[p:p + k_in]
    p += k_in
    W_out = theta[p:p + 3 * k_out].reshape(3, k_out)
    p += 3 * k_out
    b_out = theta[p:p + 3]
    p += 3
    alpha = 0.05 + 0.95 * sigmoid(theta[p])
    g = 2.0 * sigmoid(theta[p + 1])
    return W_in, b_in, W_out, b_out, alpha, g


@njit(cache=True)
def rate_forward(x, phi, rec, drive, u_sel, theta, n_feat, k_in, k_out,
                 indptr, indices, wdata, in_group, out_group, out_size,
                 mask, T, logits, rvec):
    """Run T inner steps of the rate network; writes logits (3,).
    phi must hold tanh(x)*mask on entry (zeros at game start) and is kept in sync."""
    W_in, b_in, W_out, b_out, alpha, g = rate_unpack(theta, n_feat, k_in, k_out)
    N = x.shape[0]
    for k in range(k_in):
        s = b_in[k]
        for j in range(n_feat):
            s += W_in[k, j] * u_sel[j]
        drive[k] = s
    for t in range(T):
        for i in range(N):
            s = 0.0
            for p in range(indptr[i], indptr[i + 1]):
                s += wdata[p] * phi[indices[p]]
            rec[i] = s
        for i in range(N):
            inp = 0.0
            if in_group[i] >= 0:
                inp = drive[in_group[i]]
            x[i] += alpha * (-x[i] + g * rec[i] + inp)
            phi[i] = np.tanh(x[i]) * mask[i]
    for k in range(k_out):
        rvec[k] = 0.0
    for i in range(N):
        if out_group[i] >= 0:
            rvec[out_group[i]] += phi[i]
    for k in range(k_out):
        rvec[k] /= max(out_size[k], 1)
    for a in range(3):
        s = b_out[a]
        for k in range(k_out):
            s += W_out[a, k] * rvec[k]
        logits[a] = s


@njit(cache=True)
def mlp_forward(theta, u_sel, n_feat, hidden, h, logits):
    p = 0
    for k in range(hidden):
        s = theta[p + n_feat]
        for j in range(n_feat):
            s += theta[p + j] * u_sel[j]
        h[k] = np.tanh(s)
        p += n_feat + 1
    for a in range(3):
        s = theta[p + hidden]
        for k in range(hidden):
            s += theta[p + k] * h[k]
        logits[a] = s
        p += hidden + 1


@njit(cache=True)
def argmax3(logits):
    best = 0
    for a in (1, 2):
        if logits[a] > logits[best]:
            best = a
    return best
