#!/bin/bash
# data_dir=/scratch/c00590656/vulnerability/LineVul/data/big-vul_dataset/jsonl_version
# code_dir=/scratch/c00590656/vulnerability/data-package/models/CodeBERT/code
data_dir=/scratch/lzhan011/vulnerability/LineVul/data/big-vul_dataset/
code_dir=/scratch/lzhan011/vulnerability/data-package/models/CodeBERT/code
python $code_dir/run2_flip_loni.py \
--output_folder_name=MSR \
--output_dir=./saved_models_msr_flip \
--model_type=roberta \
--tokenizer_name=microsoft/codebert-base \
--model_name_or_path=microsoft/codebert-base \
--do_test \
--train_data_file=$data_dir/train.csv \
--eval_data_file=$data_dir/train.csv \
--test_data_file=$data_dir/train.csv \
--epoch 20 \
--block_size 400 \
--train_batch_size 16 \
--eval_batch_size 1 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 1  \
--epsilon 1
