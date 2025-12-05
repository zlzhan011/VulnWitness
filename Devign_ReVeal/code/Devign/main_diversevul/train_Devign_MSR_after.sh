#!/bin/bash

fold=$1
seed=$2
savedir=models/MSR/ggnn/v$fold/$seed
echo training fold $fold seed $seed
#rm -rf $savedir
dataset_dir=/data/cs_lzhan011/data/cs_lzhan011/vulnerability/data-package/models/Devign_ReVeal/code
dataset_dir=/data1/cs_lzhan011/vulnerability/data-package/models/Devign_ReVeal/code
#exec python -u $dataset_dir/Vuld_SySe/representation_learning/api_test.py --input_dir $dataset_dir/data/MSR/full_experiment_real_data_processed --fold $fold --seed $seed --features ggnn --train 2>&1 | tee "$(basename $0)_reveal_${fold}_${seed}.log"
#exit

python -u $dataset_dir/Devign/main_diversevul/main_eval_msr_after.py --dataset DiverseVul --input_dir $dataset_dir/data/DiverseVul/full_experiment_real_data_processed --fold $fold --seed $seed \
    --model_type devign --node_tag node_features --graph_tag graph   --label_tag targets --batch_size 256  --eval_export --save_after_ggnn 2>&1 | tee "logs/$(basename $0)_ggnn_${fold}_${seed}.log"
retVal=$?
if [ $retVal -ne 0 ]; then
    echo "Error"
    exit $?
fi

#exec python -u $dataset_dir/Vuld_SySe/representation_learning/api_test.py --input_dir $dataset_dir/data/MSR/full_experiment_real_data_processed --fold $fold --seed $seed --features ggnn --train 2>&1 | tee "$(basename $0)_reveal_${fold}_${seed}.log"
