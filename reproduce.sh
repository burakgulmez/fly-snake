#!/usr/bin/env bash
# Reproduce the Snake study as reported in the paper (about 1.5-2 h on an 8-core laptop).
set -e
cd "$(dirname "$0")"
python experiments/inspect_data.py
python experiments/select_subcircuits.py
bash experiments/run_all_v2.sh          # all v2 training runs + results/v2/*.csv
python experiments/analysis.py v2_main_connectome
python experiments/export_demo.py v2
python figures/make_all.py
python paper/make_numbers.py
# The "earlier attempt" numbers in the paper come from the first design:
#   bash experiments/run_all.sh   (about 2 h; results in results/*.csv)
