#!/bin/bash

subset=all_data
data_path=/scratch/c00590656/vulnerability/LineVul/data/diversevul


# ./saved_models/multi_category_resample_V2_P50_$subset 里面存放的是resample v1版本的模型

code_path=/scratch/c00590656/vulnerability/data-package/models/CodeBERT/code/binary_category
exec python $code_path/run2_binary_category_cross_project.py \
--output_folder_name=MSR \
--output_dir=/scratch/c00590656/vulnerability/data-package/models/CodeBERT/scripts/saved_models_diversevul \
--model_type=roberta \
--tokenizer_name=microsoft/codebert-base \
--model_name_or_path=microsoft/codebert-base \
--do_eval \
--train_data_file=$data_path/train.jsonl \
--eval_data_file=$data_path/test.jsonl \
--test_data_file=$data_path/valid.jsonl \
--epoch 20 \
--block_size 400 \
--train_batch_size 32 \
--eval_batch_size 64 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 1