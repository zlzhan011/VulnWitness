#!/bin/bash

subset=all_data
data_path=/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/CodeBERT/add_after_to_before
test_data_path=/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/CodeBERT

# ./saved_models/multi_category_resample_V2_P50_$subset 里面存放的是resample v1版本的模型
# 


code_path=/data/cs_lzhan011/data/cs_lzhan011/vulnerability/data-package/models/CodeBERT/code/binary_category
exec python $code_path/run2_binary_category_cross_project.py \
--output_folder_name=$subset \
--output_dir=./saved_models/binary_category_add_after_to_before \
--model_type=roberta \
--tokenizer_name=microsoft/codebert-base \
--model_name_or_path=microsoft/codebert-base \
--do_eval \
--train_data_file=$data_path/train.csv.jsonl \
--eval_data_file=$test_data_path/test.csv.jsonl \
--test_data_file=$test_data_path/test.csv.jsonl \
--epoch 20 \
--block_size 400 \
--train_batch_size 32 \
--eval_batch_size 64 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 1 2>&1 | tee "test_$(echo $subset | sed s@/@-@g)_resample_V2_P50.log"


