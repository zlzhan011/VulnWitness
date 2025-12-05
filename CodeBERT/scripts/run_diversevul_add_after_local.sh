#!/bin/bash
data_dir=/scratch/c00590656/vulnerability/LineVul/data/diversevul
code_dir=/scratch/c00590656/vulnerability/data-package/models/CodeBERT/code
python $code_dir/run2.py \
--output_folder_name=MSR \
--output_dir=./saved_models_diversevul_add_after \
--model_type=roberta \
--tokenizer_name=microsoft/codebert-base \
--model_name_or_path=microsoft/codebert-base \
--do_train \
--train_data_file=$data_dir/train_add_after.jsonl \
--eval_data_file=$data_dir/valid_add_after.jsonl \
--test_data_file=$data_dir/test_add_after.jsonl \
--epoch 20 \
--block_size 400 \
--train_batch_size 16 \
--eval_batch_size 16 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 1
