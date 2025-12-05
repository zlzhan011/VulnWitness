import csv
import json
import os.path
import sys
import pandas as pd
from tqdm import tqdm
# 增大字段大小限制
csv.field_size_limit(sys.maxsize)

def csv_convert_jsonl(input_file, output_file):
    # 打开 CSV 文件并读取内容
    with open(input_file, mode='r', encoding='utf-8') as csv_f:
        csv_reader = csv.DictReader(csv_f)  # 使用 DictReader 将每行解析为字典
        total_rows = sum(1 for _ in open(input_file, mode='r', encoding='utf-8')) - 1  # 获取总行数（减去标题行）

        with open(output_file, mode='w', encoding='utf-8') as jsonl_f:
            for row in tqdm(csv_reader, total=total_rows, desc="Processing rows"):  # 使用 tqdm 包装 csv_reader
                row_selected_column = {}
                row_selected_column['index'] = row['index']
                row_selected_column['processed_func'] = row['processed_func']
                row_selected_column['func_after'] = row['func_after']
                row_selected_column['func_before'] = row['func_before']
                row_selected_column['target'] = row['target']
                json.dump(row, jsonl_f)  # 将字典写入 JSONL 文件
                jsonl_f.write('\n')  # 写入换行符，确保每条记录在新的一行


def csv_to_jsonl_pandas(input_file, output_file):
    print("start read")
    df = pd.read_csv(input_file)  # 快速读取 CSV
    print("start write")
    df.to_json(output_file, orient='records', lines=True, force_ascii=False)

# csv_to_jsonl_pandas('example.csv', 'example.jsonl')

if __name__ == '__main__':
    input_dir = '/work/lzhan011/vulnerability/LineVul/data/diversevul'
    input_dir = '/scratch/c00590656/vulnerability/LineVul/data/diversevul'
    input_dir = '/scratch/c00590656/vulnerability/LineVul/data/big-vul_dataset'

    # 输入和输出文件路径
    # csv_file = 'input.csv'  # 输入的 CSV 文件名
    # jsonl_file = 'output.jsonl'  # 输出的 JSONL 文件名

    # for file in os.listdir(input_dir):
    file_list = os.listdir(input_dir)
    file_list = [file for file in file_list if "flip" in file and "bak" not in file]
    file_list = ['test.csv', 'val.csv','train.csv', ]
    # file_list = ['test_flip_5_percent.csv']
    print(file_list)

    for file in  file_list:
        # csv_convert_jsonl(os.path.join(input_dir, file), os.path.join(input_dir, file[:-3]+"jsonl"))
        csv_to_jsonl_pandas(os.path.join(input_dir, file), os.path.join(input_dir, 'jsonl_version', file[:-3] + "jsonl"))
