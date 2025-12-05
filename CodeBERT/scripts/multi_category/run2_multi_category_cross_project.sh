#!/bin/bash

subset=fold_0_dataset
data_path=/data/cs_lzhan011/data/cs_lzhan011/vulnerability/data-package/cross_sections/code/data/subsets/multi_category_cross_project/
data_path=$data_path$subset


# ./saved_models/multi_category_resample_V2_P50_$subset 里面存放的是resample v1版本的模型
# 


code_path=/data/cs_lzhan011/data/cs_lzhan011/vulnerability/data-package/models/CodeBERT/code
exec python $code_path/run2_multi_category_cross_project.py \
--output_folder_name=$subset \
--output_dir=./saved_models/multi_category_resample_V2_P50_$subset \
--model_type=roberta \
--tokenizer_name=microsoft/codebert-base \
--model_name_or_path=microsoft/codebert-base \
--do_train \
--train_data_file=$data_path/CodeBERT/train.csv.jsonl \
--eval_data_file=$data_path/CodeBERT/valid.csv.jsonl \
--test_data_file=$data_path/CodeBERT/test.csv.jsonl \
--epoch 20 \
--block_size 400 \
--train_batch_size 32 \
--eval_batch_size 64 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 1 2>&1 | tee "train_$(echo $subset | sed s@/@-@g)_resample_V2_P50.log"







subset=fold_1_dataset
data_path=/data/cs_lzhan011/data/cs_lzhan011/vulnerability/data-package/cross_sections/code/data/subsets/multi_category_cross_project/
data_path=$data_path$subset

code_path=/data/cs_lzhan011/data/cs_lzhan011/vulnerability/data-package/models/CodeBERT/code
exec python $code_path/run2_multi_category_cross_project.py \
--output_folder_name=$subset \
--output_dir=./saved_models/multi_category_resample_V2_P50_$subset \
--model_type=roberta \
--tokenizer_name=microsoft/codebert-base \
--model_name_or_path=microsoft/codebert-base \
--do_train \
--train_data_file=$data_path/CodeBERT/train.csv.jsonl \
--eval_data_file=$data_path/CodeBERT/valid.csv.jsonl \
--test_data_file=$data_path/CodeBERT/test.csv.jsonl \
--epoch 20 \
--block_size 400 \
--train_batch_size 32 \
--eval_batch_size 64 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 1 2>&1 | tee "train_$(echo $subset | sed s@/@-@g)_resample_V2_P50.log"









subset=fold_2_dataset
data_path=/data/cs_lzhan011/data/cs_lzhan011/vulnerability/data-package/cross_sections/code/data/subsets/multi_category_cross_project/
data_path=$data_path$subset

code_path=/data/cs_lzhan011/data/cs_lzhan011/vulnerability/data-package/models/CodeBERT/code
exec python $code_path/run2_multi_category_cross_project.py \
--output_folder_name=$subset \
--output_dir=./saved_models/multi_category_resample_V2_P50_$subset \
--model_type=roberta \
--tokenizer_name=microsoft/codebert-base \
--model_name_or_path=microsoft/codebert-base \
--do_train \
--train_data_file=$data_path/CodeBERT/train.csv.jsonl \
--eval_data_file=$data_path/CodeBERT/valid.csv.jsonl \
--test_data_file=$data_path/CodeBERT/test.csv.jsonl \
--epoch 20 \
--block_size 400 \
--train_batch_size 32 \
--eval_batch_size 64 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 1 2>&1 | tee "train_$(echo $subset | sed s@/@-@g)_resample_V2_P50.log"





subset=fold_3_dataset
data_path=/data/cs_lzhan011/data/cs_lzhan011/vulnerability/data-package/cross_sections/code/data/subsets/multi_category_cross_project/
data_path=$data_path$subset

code_path=/data/cs_lzhan011/data/cs_lzhan011/vulnerability/data-package/models/CodeBERT/code
exec python $code_path/run2_multi_category_cross_project.py \
--output_folder_name=$subset \
--output_dir=./saved_models/multi_category_resample_V2_P50_$subset \
--model_type=roberta \
--tokenizer_name=microsoft/codebert-base \
--model_name_or_path=microsoft/codebert-base \
--do_train \
--train_data_file=$data_path/CodeBERT/train.csv.jsonl \
--eval_data_file=$data_path/CodeBERT/valid.csv.jsonl \
--test_data_file=$data_path/CodeBERT/test.csv.jsonl \
--epoch 20 \
--block_size 400 \
--train_batch_size 32 \
--eval_batch_size 64 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 1 2>&1 | tee "train_$(echo $subset | sed s@/@-@g)_resample_V2_P50.log"





subset=fold_4_dataset
data_path=/data/cs_lzhan011/data/cs_lzhan011/vulnerability/data-package/cross_sections/code/data/subsets/multi_category_cross_project/
data_path=$data_path$subset

code_path=/data/cs_lzhan011/data/cs_lzhan011/vulnerability/data-package/models/CodeBERT/code
exec python $code_path/run2_multi_category_cross_project.py \
--output_folder_name=$subset \
--output_dir=./saved_models/multi_category_resample_V2_P50_$subset \
--model_type=roberta \
--tokenizer_name=microsoft/codebert-base \
--model_name_or_path=microsoft/codebert-base \
--do_train \
--train_data_file=$data_path/CodeBERT/train.csv.jsonl \
--eval_data_file=$data_path/CodeBERT/valid.csv.jsonl \
--test_data_file=$data_path/CodeBERT/test.csv.jsonl \
--epoch 20 \
--block_size 400 \
--train_batch_size 32 \
--eval_batch_size 64 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 1 2>&1 | tee "train_$(echo $subset | sed s@/@-@g)_resample_V2_P50.log"






subset=fold_5_dataset
data_path=/data/cs_lzhan011/data/cs_lzhan011/vulnerability/data-package/cross_sections/code/data/subsets/multi_category_cross_project/
data_path=$data_path$subset

code_path=/data/cs_lzhan011/data/cs_lzhan011/vulnerability/data-package/models/CodeBERT/code
exec python $code_path/run2_multi_category_cross_project.py \
--output_folder_name=$subset \
--output_dir=./saved_models/multi_category_resample_V2_P50_$subset \
--model_type=roberta \
--tokenizer_name=microsoft/codebert-base \
--model_name_or_path=microsoft/codebert-base \
--do_train \
--train_data_file=$data_path/CodeBERT/train.csv.jsonl \
--eval_data_file=$data_path/CodeBERT/valid.csv.jsonl \
--test_data_file=$data_path/CodeBERT/test.csv.jsonl \
--epoch 20 \
--block_size 400 \
--train_batch_size 32 \
--eval_batch_size 64 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--evaluate_during_training \
--seed 1 2>&1 | tee "train_$(echo $subset | sed s@/@-@g)_resample_V2_P50.log"