#!/bin/bash

# $1 strategy, $2 memory size

for SEED in 3 4;
do
    python ../experiments/main.py strategy="$1" +best_configs=split_tinyimagenet/$1 \
        strategy.mem_size=$2 experiment.seed=$SEED experiment.save_models=true \
        evaluation=parallel experiment=split_tinyimagenet deploy=default
done
