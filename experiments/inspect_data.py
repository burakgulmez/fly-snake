"""Data inspection numbers used in the paper -> results/data_inspection.json"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import pandas as pd

RAW = os.path.join(ROOT, "data", "raw")
n = pd.read_csv(os.path.join(RAW, "neurons.csv.gz"))
con = pd.read_csv(os.path.join(RAW, "connections_princeton.csv.gz"))
pairs = con.groupby(["pre_root_id", "post_root_id"]).syn_count.sum()
m = con.merge(n[["root_id", "nt_type"]].rename(columns={"root_id": "pre_root_id", "nt_type": "nt_neuron"}),
              on="pre_root_id", how="left")
w = m.syn_count
out = dict(n_neurons=int(len(n)), n_connection_rows=int(len(con)), n_pairs=int(len(pairs)),
           min_pair_syn=int(pairs.min()), n_synapses_in_filtered_table=int(con.syn_count.sum()),
           nt_agree_syn_frac=float((w * (m.nt_type == m.nt_neuron)).sum() / w.sum()),
           nt_neuron_missing_syn_frac=float((w * m.nt_neuron.isna()).sum() / w.sum()),
           nt_conflict_syn_frac=float((w * ((m.nt_type != m.nt_neuron) & m.nt_neuron.notna())).sum() / w.sum()),
           pre_neurons_multiple_conn_nt=int((con.groupby("pre_root_id").nt_type.nunique() > 1).sum()))
json.dump(out, open(os.path.join(ROOT, "results", "data_inspection.json"), "w"), indent=1)
print(out)
