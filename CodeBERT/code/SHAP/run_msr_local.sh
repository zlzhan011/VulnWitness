#!/bin/bash
data_dir=/scratch/c00590656/vulnerability/LineVul/data/big-vul_dataset/jsonl_version
code_dir=/scratch/c00590656/vulnerability/data-package/models/CodeBERT/code/SHAP
python $code_dir/run2_SHAP_CODEBERT.py \
--output_folder_name=MSR \
--output_dir=/scratch/c00590656/vulnerability/data-package/models/CodeBERT/scripts/saved_models_msr_processed_func \
--model_type=roberta \
--tokenizer_name=microsoft/codebert-base \
--model_name_or_path=microsoft/codebert-base \
--do_test  \
--do_shap \
--train_data_file=$data_dir/train.jsonl \
--eval_data_file=$data_dir/test.jsonl \
--test_data_file=$data_dir/val.jsonl \
--epoch 20 \
--block_size 400 \
--train_batch_size 16 \
--eval_batch_size 16 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 1
