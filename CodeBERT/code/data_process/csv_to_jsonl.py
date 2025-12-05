
import  pandas as pd
import os
# c_root_root = '/data/cs_lzhan011/data/cs_lzhan011/vulnerability/data-package/cross_sections/code/data/subsets/multi_category_cross_project/'
# file_name = 'holdout.csv'
# for dir in ['fold_0_holdout', 'fold_1_holdout', 'fold_2_holdout', 'fold_3_holdout', 'fold_4_holdout', 'fold_5_holdout']:
#     c_root = os.path.join(c_root_root, dir)
#     for file_name in ['holdout.csv']:
#         file_path = os.path.join(c_root, file_name)
#         df = pd.read_csv(file_path)
#         # df = df.head(10000)
#         # 将DataFrame转换为JSONL格式
#         jsonl_string = df.to_json(orient='records', lines=True)
#
#         if os.path.isdir(os.path.join(c_root, 'CodeBERT')):
#             pass
#         else:
#             os.mkdir(os.path.join(c_root, 'CodeBERT'))
#         output_path = os.path.join(c_root, 'CodeBERT', file_name+".jsonl")
#         with open(output_path, 'w') as file:
#             file.write(jsonl_string)





c_root_root = '/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/add_after_to_before'
c_output = '/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/CodeBERT/add_after_to_before'
file_name = 'train.csv'
for dir in ['']:
    c_root = c_root_root
    for file_name in [ 'train.csv']: # 'train.csv', 'test.csv', valid.csv
        file_path = os.path.join(c_root, file_name)
        df = pd.read_csv(file_path)
        # df = df.head(10000)
        # 将DataFrame转换为JSONL格式
        jsonl_string = df.to_json(orient='records', lines=True)

        # if os.path.isdir(os.path.join(c_output, 'CodeBERT')):
        #     pass
        # else:
        #     os.mkdir(os.path.join(c_output, 'CodeBERT'))
        output_path = os.path.join(c_output, file_name+".jsonl")
        with open(output_path, 'w') as file:
            file.write(jsonl_string)