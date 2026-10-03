#!/usr/bin/env bash
# Runs every experiment of the paper (skips runs that already have summary.json).
set -e
cd "$(dirname "$0")/.."
run() { exp=$1; seed=$2; shift 2
  if [ -f results/runs/$exp/seed$seed/summary.json ]; then return; fi
  echo "[$(date +%T)] $exp seed $seed"; python experiments/run.py --exp $exp --seed $seed --set "$@" | tail -1; }
SEEDS="0 1 2 3 4"
for m in random greedy floodfill; do run base_$m 0 model=$m; done          # deterministic agents
for s in $SEEDS; do run main_mlp $s model=mlp; done
for s in $SEEDS; do
  for m in connectome degree_shuffle erdos_renyi sign_shuffle no_recurrence; do run main_$m $s model=$m; done
done
for s in $SEEDS; do
  run abl_feat_food_danger $s feature_set=food_danger
  run abl_feat_food_danger_space $s feature_set=food_danger_space
  run abl_fitness_food_only $s fitness.fill=0 fitness.win=0 fitness.timeout=0
done
for s in $SEEDS; do run scale_A800 $s circuit=A_vpn_dn_800; done
python experiments/aggregate.py
