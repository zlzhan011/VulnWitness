#!/bin/bash

fold=$1
seed=$2
echo evaling fold $fold seed $seed
dataset_dir=/data/cs_lzhan011/data/cs_lzhan011/vulnerability/data-package/models/Devign_ReVeal/code

python -u $dataset_dir/Vuld_SySe/representation_learning/api_test.py \
    --input_dir data/MSR/full_experiment_real_data_processed \
    --fold $fold --seed $seed --features ggnn \
    --eval_export
