"""Python (numba) vs JS (demo/sim.js): identical action sequences for 20 seeded games."""
import glob
import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from src.brain.build import build_spec
from src.eval.rollout import trace_game
from src.export.export_weights import export_model

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


RUNS = ["results/runs/main_connectome/seed0", "results/runs/v2_main_connectome/seed0",
        "results/runs/v2_main_degree_shuffle/seed0"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("rel", RUNS)
def test_python_js_identical_actions(tmp_path, rel):
    run = os.path.join(ROOT, rel)
    if not os.path.exists(os.path.join(run, "theta.npy")):
        pytest.skip("run not available")
    cfg = json.load(open(os.path.join(run, "config.json")))
    theta = np.load(os.path.join(run, "theta.npy"))
    seed = int(os.path.basename(run)[4:])
    spec, _, _ = build_spec(cfg, control_seed=seed)
    m = export_model(cfg, theta, control_seed=seed)
    wpath = tmp_path / "w.json"
    wpath.write_text(json.dumps(m))
    seeds = list(range(2_000_000_000, 2_000_000_020))
    script = f"""
const S=require({json.dumps(os.path.join(ROOT, 'demo', 'sim.js'))});
const m=JSON.parse(require('fs').readFileSync({json.dumps(str(wpath))}));
const net=new S.RateNet(m); const out=[];
for (const s of {json.dumps(seeds)}) out.push(S.playGame(net,s,m.grid,m.timeout_k).actions);
console.log(JSON.stringify(out));
"""
    res = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    js = json.loads(res.stdout)
    for s, ja in zip(seeds, js):
        _, pa, _ = trace_game(spec, theta, s, cfg["grid"], cfg["timeout_k"])
        assert list(pa) == ja, f"seed {s}: first diff at {next(i for i,(a,b) in enumerate(zip(pa,ja)) if a!=b) if len(pa)==len(ja) else 'length'}"
