# Data

The FlyWire tables are not stored in this repository.

## Download (FlyWire Codex, FAFB v783)

1. Open <https://codex.flywire.ai>, choose the dataset **FAFB v783** and go to the
   *Download* page (a free login and acceptance of the FlyWire terms are required).
2. Download these files into `data/raw/` (keep them gzipped):

| Codex name | File |
|---|---|
| Connections (Filtered) | `connections_princeton.csv.gz` |
| Neurotransmitter Type Predictions | `neurons.csv.gz` |
| Classification / Hierarchical Annotations | `classification.csv.gz` |
| Cell Types | `consolidated_cell_types.csv.gz` |
| Proofread Cell Names And Groups | `names.csv.gz` |

The Codex files are updated over time, so your copies may differ slightly from the ones
used here. [`raw/MANIFEST.md`](raw/MANIFEST.md) lists the sizes and SHA-256 checksums of
the files behind the published results.

## Processed files

- `processed/fafb783_graph.npz` (not committed): neuron-pair graph with synapse counts summed
  over brain regions. It is built automatically from `data/raw/` the first time it is needed.
- The small extracted circuit used by the experiments is committed, so the demo, the tests
  and the trained results work without downloading anything.
