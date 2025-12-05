#!/bin/bash

fold=$1
seed=$2
savedir=models/reveal/ggnn/v$fold/$seed
echo training fold $fold seed $seed
#rm -rf $savedir
dataset_dir=/data/cs_lzhan011/data/cs_lzhan011/vulnerability/data-package/models/Devign_ReVeal/code

python -u $dataset_dir/Devign/main_reveal_reveal.py --dataset ReVeal --input_dir $dataset_dir/data/ReVeal/full_experiment_real_data_processed --fold $fold --seed $seed \
    --model_type ggnn --node_tag node_features --graph_tag graph --label_tag targets --batch_size 256 --train --eval_export --save_after_ggnn 2>&1 | tee "$(basename $0)_ggnn_${fold}_${seed}.log"
retVal=$?
if [ $retVal -ne 0 ]; then
    echo "Error"
    exit $?
fi
