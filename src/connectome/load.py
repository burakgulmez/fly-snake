"""Load FlyWire FAFB v783 Codex tables and build a pair-aggregated graph cache."""
import os
import numpy as np
import pandas as pd

# Raw FlyWire tables: data/raw. Graph cache and extracted subcircuits: data/processed.
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW = os.path.join(REPO, "data", "raw")
SHARED_PROC = os.path.join(REPO, "data", "processed")
PROC = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
CACHE = os.path.join(SHARED_PROC, "fafb783_graph.npz")

# Sign rule (ASSUMPTION, documented in paper): ACh excitatory; GABA and glutamate
# inhibitory; monoamines (DA, SER, OCT) are modulatory and mapped to `modulatory_sign`.
NT_SIGN = {"ACH": 1.0, "GABA": -1.0, "GLUT": -1.0, "DA": None, "SER": None, "OCT": None}


def load_meta():
    n = pd.read_csv(os.path.join(RAW, "neurons.csv.gz"))
    c = pd.read_csv(os.path.join(RAW, "classification.csv.gz"))
    t = pd.read_csv(os.path.join(RAW, "consolidated_cell_types.csv.gz"))
    nm = pd.read_csv(os.path.join(RAW, "names.csv.gz"))
    meta = (c.merge(n[["root_id", "nt_type"]].rename(columns={"nt_type": "nt_neuron"}), on="root_id", how="left")
             .merge(t[["root_id", "primary_type"]], on="root_id", how="left")
             .merge(nm[["root_id", "name"]], on="root_id", how="left"))
    return meta.sort_values("root_id").reset_index(drop=True)


def build_graph(force=False):
    """Returns dict: root_ids (sorted), pre, post (indices), syn (int), nt (per-pre str codes)."""
    if os.path.exists(CACHE) and not force:
        z = np.load(CACHE, allow_pickle=False)
        return {k: z[k] for k in z.files}
    os.makedirs(SHARED_PROC, exist_ok=True)
    os.makedirs(PROC, exist_ok=True)
    con = pd.read_csv(os.path.join(RAW, "connections_princeton.csv.gz"))
    # per-connection NT is constant per presynaptic neuron (checked in inspection)
    pre_nt = con.groupby("pre_root_id").nt_type.first()
    pairs = con.groupby(["pre_root_id", "post_root_id"], sort=False).syn_count.sum().reset_index()
    meta = load_meta()
    ids = meta.root_id.values
    pre = np.searchsorted(ids, pairs.pre_root_id.values)
    post = np.searchsorted(ids, pairs.post_root_id.values)
    assert np.all(ids[pre] == pairs.pre_root_id.values) and np.all(ids[post] == pairs.post_root_id.values)
    nt = pd.Series(ids).map(pre_nt).fillna("").values.astype("U4")
    out = dict(root_ids=ids, pre=pre.astype(np.int64), post=post.astype(np.int64),
               syn=pairs.syn_count.values.astype(np.int64), nt=nt)
    np.savez_compressed(CACHE, **out)
    return out
