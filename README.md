# Snake with a fly visual-to-motor circuit

One of three small hobby studies in which a circuit from the fruit fly connectome plays a game. The other two: [Flappy Bird](https://github.com/burakgulmez/fly-flappybird), [Pong](https://github.com/burakgulmez/fly-pong).

A fixed network wired like **133 neurons of the fruit fly brain** plays Snake on a 10×10 board:
- **Inputs:** 52 visual projection neurons (VPNs)
- **Outputs:** 63 descending neurons (DNs)
- **In between:** 18 neurons on paths of at most two steps between them

Only 157 interface numbers are trained.

**[Open the demo](demo/index.html)**: watch the network play, see its neurons light up, switch to a
shuffled wiring, or silence the most active neurons.

## Results (500 unseen test games per trained network)

| Network (10 training runs each) | Food per game |
|---|---|
| FlyWire circuit | **27.3 ± 3.6** |
| same circuit, connections shuffled (degrees kept) | 26.9 ± 2.4 |
| random graph, same size | 26.0 ± 2.0 |
| FlyWire circuit, signs shuffled | 25.7 ± 3.7 |
| no internal connections | 28.9 ± 2.9 |
| greedy hand-written rule | 18.4 |
| random moves | 0.2 |

The network learns to chase food and never hits walls. About half of its games end by
running into its own body. In this simple setup the particular fly wiring gave **no visible
advantage** over shuffled or random wiring. Details are in [`paper/main.tex`](paper/main.tex).

## How it works

- **Circuit** (`src/connectome/`, `experiments/select_subcircuits.py`): VPN → DN paths of at most two
  steps over pairs with ≥ 200 synapses; all connections with ≥ 5 synapses between the kept neurons.
  Weight = sign × log(1 + synapses). Acetylcholine is positive; GABA and glutamate are negative.
- **Game** (`src/snake/engine.py`): numba, deterministic 32-bit xorshift food placement, relative
  actions (left / straight / right).
- **Sensors** (15): food offset and direction, immediate danger, distance to the nearest obstacle in
  three directions, diagonal cells.
- **Network** (`src/brain/policies.py`): rate model `x ← x + a(−x + g·W·tanh(x) + I)`, 10 updates per
  step; 8 input groups and 8 output groups.
- **Training** (`src/train/es.py`, `experiments/run.py`): CMA-ES, 4M game steps per run (~70 s).

## Reproduce

```bash
# FlyWire tables in data/raw (see data/README.md)
bash reproduce.sh          # ~1.5-2 h: circuits, all training runs, analyses, figures, paper numbers
python -m pytest tests     # game rules and Python/JS equivalence (needs Node.js for the JS test)
```

Main configuration: `experiments/configs/base_v2.yaml`. Results: `results/v2/` and
`results/runs/v2_*`.

The files without `v2` (`experiments/configs/base.yaml`, `experiments/run_all.sh`, `results/*.csv`,
`results/runs/main_*`, …) belong to a first attempt: a 278-neuron circuit, flood-fill sensors and
half the budget, which scored 17.1 ± 9.0. The paper mentions it briefly.

## Layout

```
src/            game engine (numba), circuit extraction, network
experiments/    training and analysis scripts
results/        logged results (CSV / JSON) used by the paper
figures/        make_all.py -> figures
paper/          main.tex + make_numbers.py (every number is generated from results/)
arxiv/          self-contained arXiv submission package
demo/           static HTML/JS demo (index.html, sim.js, weights.js) and a short video
tests/          pytest (game rules, Python/JS equivalence)
tools/          reference fetcher and verifier (OpenAlex, Crossref), arXiv packaging, video recorder
data/           where the FlyWire tables go (see data/README.md)
```

Install with `pip install -r requirements.txt` (Python 3.10+). Node.js is only needed for the
Python/JS equivalence test and for recording the demo video.

## Data and credit

The connectome data come from the FlyWire project (FAFB v783) via the
[FlyWire Codex](https://codex.flywire.ai) download page and are not included here.
Please follow the FlyWire citation guidelines when using the data.

This is not a model of how flies think, and flies do not play Snake. It is a small,
reproducible experiment with real connectome data.

## License

Code: MIT (see `LICENSE`). The FlyWire data are subject to their own terms.
