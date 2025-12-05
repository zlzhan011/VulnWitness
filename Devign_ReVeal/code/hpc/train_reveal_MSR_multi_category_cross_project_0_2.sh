#!/bin/bash

fold=$1
seed=$2
savedir=models/MSR/ggnn/v$fold/$seed
echo training fold $fold seed $seed
rm -rf $savedir
dataset_dir=/data/cs_lzhan011/data/cs_lzhan011/vulnerability/data-package/models/Devign_ReVeal/code
#exec python -u $dataset_dir/Vuld_SySe/representation_learning/api_test.py --input_dir $dataset_dir/data/MSR/full_experiment_real_data_processed --fold $fold --seed $seed --features ggnn --train 2>&1 | tee "$(basename $0)_reveal_${fold}_${seed}_resample_v2_P50.log"
#exit
fold_dir=fold_0_dataset
python -u $dataset_dir/Devign/main_eval_msr_multi_category_cross_project.py --dataset MSR --input_dir $dataset_dir/data/MSR/full_experiment_real_data_processed --fold $fold --seed $seed \
    --model_type ggnn --node_tag node_features --graph_tag graph --label_tag targets --fold_dir $fold_dir  --batch_size 256 --train --eval_export --save_after_ggnn 2>&1 | tee "$(basename $0)_reveal_${fold}_${seed}_${fold_dir}_resample_v2_P50.log"




#
#
#fold_dir=fold_1_dataset
#python -u $dataset_dir/Devign/main_eval_msr_multi_category_cross_project.py --dataset MSR --input_dir $dataset_dir/data/MSR/full_experiment_real_data_processed --fold $fold --seed $seed \
#    --model_type ggnn --node_tag node_features --graph_tag graph --label_tag targets --fold_dir $fold_dir  --batch_size 256 --train --eval_export --save_after_ggnn 2>&1 | tee "$(basename $0)_reveal_${fold}_${seed}_${fold_dir}_resample_v2_P50.log"
#
#
#
#fold_dir=fold_2_dataset
#python -u $dataset_dir/Devign/main_eval_msr_multi_category_cross_project.py --dataset MSR --input_dir $dataset_dir/data/MSR/full_experiment_real_data_processed --fold $fold --seed $seed \
#    --model_type ggnn --node_tag node_features --graph_tag graph --label_tag targets --fold_dir $fold_dir  --batch_size 256 --train --eval_export --save_after_ggnn 2>&1 | tee "$(basename $0)_reveal_${fold}_${seed}_${fold_dir}_resample_v2_P50.log"

#retVal=$?
#if [ $retVal -ne 0 ]; then
#    echo "Error"
#    exit $?
#fi
#exec python -u $dataset_dir/Vuld_SySe/representation_learning/api_test.py --input_dir $dataset_dir/data/MSR/full_experiment_real_data_processed --fold $fold --seed $seed --features ggnn --train 2>&1 | tee "$(basename $0)_reveal_${fold}_${seed}_resample_v2_P50.log"
