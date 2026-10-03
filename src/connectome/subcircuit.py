"""Subcircuit selection, sign/weight rule, spectral normalisation, I/O groups."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs

from src.connectome.load import build_graph, load_meta, NT_SIGN


def _adj(g, N, thr):
    m = g["syn"] >= thr
    return sp.csr_matrix((np.ones(m.sum()), (g["pre"][m], g["post"][m])), shape=(N, N))


def _msbfs(A, src, L):
    """Multi-source BFS along edge direction of A (rows=pre). Distances capped at 99."""
    N = A.shape[0]
    d = np.full(N, 99)
    d[src] = 0
    fr = np.zeros(N)
    fr[src] = 1
    for k in range(1, L + 1):
        nxt = (fr @ A) > 0
        nxt &= d == 99
        d[nxt] = k
        fr = nxt.astype(float)
    return d


def select_nodes(cfg, g, meta):
    N = len(meta)
    kind = cfg["kind"]
    pt = meta.primary_type.fillna("")
    if kind == "path":
        src = np.where(meta.super_class.isin(cfg["sources_super_class"]).values)[0]
        dst = np.where(meta.super_class.isin(cfg["targets_super_class"]).values)[0]
        A = _adj(g, N, cfg["path_syn_threshold"])
        dS = _msbfs(A, src, cfg["max_depth"])
        dD = _msbfs(A.T.tocsr(), dst, cfg["max_depth"])
        keep = (dS + dD) <= cfg["max_depth"]
        nodes = np.where(keep)[0]
        inputs = np.intersect1d(nodes, src)
        outputs = np.setdiff1d(np.intersect1d(nodes, dst), inputs)
    elif kind == "types":
        core = np.where(pt.str.match(cfg["core_type_regex"]).values)[0]
        inputs = np.where(pt.str.match(cfg["input_type_regex"]).values)[0]
        outpre = np.where(pt.str.match(cfg["output_type_regex"]).values)[0]
        A = sp.csr_matrix((g["syn"], (g["pre"], g["post"])), shape=(N, N))
        dn = np.where(meta.super_class.isin(cfg["extra_output_super_class"]).values)[0]
        strong = dn[(A[outpre][:, dn].sum(0).A1) >= cfg["extra_output_min_syn"]]
        nodes = np.unique(np.concatenate([core, inputs, outpre, strong]))
        outputs = np.setdiff1d(np.unique(np.concatenate([outpre, strong])), inputs)
    else:
        raise ValueError(kind)
    return nodes, inputs, outputs


def build_subcircuit(cfg):
    g = build_graph()
    meta = load_meta()
    N = len(meta)
    nodes, inputs, outputs = select_nodes(cfg, g, meta)
    pos = -np.ones(N, np.int64)
    pos[nodes] = np.arange(len(nodes))
    m = (g["syn"] >= cfg["edge_syn_threshold"]) & (pos[g["pre"]] >= 0) & (pos[g["post"]] >= 0)
    pre, post, syn = pos[g["pre"][m]], pos[g["post"][m]], g["syn"][m]
    nt = g["nt"][g["pre"][m]]
    sign = np.array([NT_SIGN[x] if NT_SIGN.get(x) is not None else cfg["modulatory_sign"] for x in nt], float)
    mag = np.log1p(syn) if cfg["weight_fn"] == "log1p" else syn / syn.max()
    n = len(nodes)
    # W[post, pre]: row i collects input to neuron i
    W = sp.csr_matrix((sign * mag, (post, pre)), shape=(n, n))
    W.sum_duplicates()
    W.sort_indices()
    sub_meta = meta.iloc[nodes].reset_index(drop=True)
    return W, sub_meta, pos[inputs], pos[outputs], nt, syn


def spectral_radius(W):
    if W.nnz == 0:
        return 0.0
    if W.shape[0] < 800:
        return float(np.abs(np.linalg.eigvals(W.toarray())).max())
    v = eigs(W.astype(float), k=4, which="LM", return_eigenvectors=False, maxiter=20000, tol=1e-10)
    return float(np.abs(v).max())


def make_groups(sub_meta_side, sub_meta_type, root_ids, idx, k):
    """Deterministic grouping: sort by (side, primary_type, root_id), split into k chunks."""
    idx = np.asarray(idx)
    order = np.lexsort((root_ids[idx], sub_meta_type[idx].astype(str), sub_meta_side[idx].astype(str)))
    groups = -np.ones(len(root_ids), np.int64)
    for gi, chunk in enumerate(np.array_split(idx[order], k)):
        groups[chunk] = gi
    return groups


def stats(W, in_local, out_local):
    n = W.shape[0]
    Wd = W.tocsr()
    indeg = np.diff(Wd.indptr)
    outdeg = np.diff(Wd.tocsc().indptr)
    B = (abs(Wd) > 0).astype(np.int8)
    recip = (B.multiply(B.T)).nnz / max(B.nnz, 1)
    fr = np.zeros(n)
    fr[in_local] = 1
    reach = np.zeros(n, bool)
    for _ in range(3):
        fr = (B @ fr) > 0  # B[post,pre]: pre -> post
        reach |= fr
        fr = fr.astype(float)
    return dict(neurons=n, edges=int(Wd.nnz), density=Wd.nnz / n / n,
                frac_exc_edges=(Wd.data > 0).sum() / max(Wd.nnz, 1),
                frac_exc_weight=Wd.data[Wd.data > 0].sum() / max(np.abs(Wd.data).sum(), 1e-12),
                in_deg_mean=indeg.mean(), in_deg_max=int(indeg.max()), out_deg_max=int(outdeg.max()),
                reciprocity=recip, n_input=len(in_local), n_output=len(out_local),
                frac_outputs_reachable_3hops=float(reach[out_local].mean()) if len(out_local) else 0.0,
                spectral_radius_raw=spectral_radius(Wd))
