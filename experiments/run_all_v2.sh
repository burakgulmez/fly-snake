#!/usr/bin/env bash
# Version-2 experiments (A-130, v2 sensors, food/survival fitness, 4M steps/run).
set -e
cd "$(dirname "$0")/.."
run() { exp=$1; seed=$2; shift 2
  if [ -f results/runs/$exp/seed$seed/summary.json ]; then return; fi
  echo "[$(date +%T)] $exp seed $seed"; python experiments/run.py --base base_v2 --exp $exp --seed $seed --set max_train_steps=4000000 "$@" | tail -1; }
for m in random greedy floodfill; do run v2_base_$m 0 model=$m; done
for s in 0 1 2 3 4 5 6 7 8 9; do run v2_main_mlp $s model=mlp; done
for s in 0 1 2 3 4 5 6 7 8 9; do
  for m in connectome degree_shuffle erdos_renyi sign_shuffle no_recurrence; do run v2_main_$m $s model=$m; done
done
for s in 0 1 2 3 4; do
  run v2_abl_no_rays $s feature_set=v2_no_rays
  run v2_scale_A70 $s circuit=A_vpn_dn_70
  run v2_scale_A280 $s circuit=A_vpn_dn_200
done
python experiments/aggregate.py v2
