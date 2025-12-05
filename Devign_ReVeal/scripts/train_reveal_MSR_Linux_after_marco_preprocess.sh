#!/bin/bash

fold=MSR_Linux_after_marco_preprocess
seed=1234
savedir=models/MSR_Linux_after_marco_preprocess/ggnn/v$fold/$seed
echo training fold $fold seed $seed

dataset_dir=/scratch/c00590656/vulnerability/data-package/models/Devign_ReVeal/code
python -u /scratch/c00590656/vulnerability/data-package/models/Devign_ReVeal/code/Devign/main_reveal_linux_subset.py --dataset MSR_Linux_after_marco_preprocess --input_dir ${dataset_dir}/data/MSR_Linux_after_marco_preprocess/full_experiment_real_data_processed --fold $fold --seed $seed \
    --model_type ggnn --node_tag node_features --graph_tag graph --label_tag targets --batch_size 256 --train --eval_export --save_after_ggnn 2>&1 | tee "$(basename $0)_ggnn_${fold}_${seed}.log"
# retVal=$?
# if [ $retVal -ne 0 ]; then
#     echo "Error"
#     exit $?
# fi
# exec python -u Vuld_SySe/representation_learning/api_test.py --input_dir data/MSR_Linux_after_marco_preprocess/full_experiment_real_data_processed --fold $fold --seed $seed --features ggnn --train 2>&1 | tee "$(basename $0)_reveal_${fold}_${seed}.log"
