"""Build all subcircuit candidates -> data/processed/<name>.npz + results/subcircuit_stats.csv"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd
import yaml

from src.connectome.subcircuit import build_subcircuit, stats

cfg = yaml.safe_load(open(os.path.join(ROOT, "experiments/configs/subcircuits.yaml")))
rows = []
for name, c in cfg["candidates"].items():
    c = {**cfg["common"], **c}
    W, sm, inp, outp, nt, syn = build_subcircuit(c)
    s = stats(W, inp, outp)
    s["name"] = name
    s["nt_counts_edges"] = json.dumps(pd.Series(nt).value_counts().to_dict())
    s["super_class_counts"] = json.dumps(sm.super_class.value_counts().to_dict())
    rows.append(s)
    np.savez_compressed(os.path.join(ROOT, f"data/processed/{name}.npz"), indptr=W.indptr, indices=W.indices,
                        data=W.data, n=W.shape[0], inputs=inp, outputs=outp, root_ids=sm.root_id.values,
                        side=sm.side.fillna("").values.astype("U8"),
                        primary_type=sm.primary_type.fillna("").values.astype("U32"),
                        super_class=sm.super_class.fillna("").values.astype("U24"), config=json.dumps(c))
    print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items() if "counts" not in k})
    print("   NT(edges):", s["nt_counts_edges"], "\n   classes:", s["super_class_counts"])
pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results/subcircuit_stats.csv"), index=False)
