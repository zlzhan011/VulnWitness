import os
import  pandas as pd
import torch



if __name__ == '__main__':
    c_dir_input = '/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/only_paired_func'
    c_dir_output = '/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/CodeBERT'
    c_dir_output = os.path.join(c_dir_output, 'only_paired_func')
    files = ['train.csv', 'test.csv', 'valid.csv']
    for file in files:
        file_path = os.path.join(c_dir_input, file)
        df = pd.read_csv(file_path)
        df.to_json(os.path.join(c_dir_output, file+".jsonl"), orient='records', lines=True)

