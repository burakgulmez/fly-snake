"""Export a trained rate model to JSON for the browser demo."""
import json

import numpy as np

from src.brain.build import build_spec




def export_model(cfg, theta, control_seed=0, extra=None):
    spec, D, info = build_spec(cfg, control_seed=control_seed)
    m = dict(model=cfg["model"], circuit=cfg["circuit"], n=int(spec.N), indptr=spec.indptr.tolist(),
             indices=spec.indices.tolist(), data=[float(v) for v in spec.wdata],
             in_group=spec.in_group.tolist(), out_group=spec.out_group.tolist(), k_in=int(spec.k_in),
             k_out=int(spec.k_out), T=int(spec.T), feat_idx=spec.feat_idx.tolist(),
             theta=[float(v) for v in np.asarray(theta)], grid=int(cfg["grid"]), feat_version=int(spec.feat_version), timeout_k=int(cfg["timeout_k"]))
    if extra:
        m.update(extra)
    return m
