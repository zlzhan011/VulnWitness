import os
import shutil

import pandas as pd
from scipy.stats import trim1


def write_c(project_dir, file_name, index, func):
    if not os.path.exists(os.path.join(project_dir,str(index))):
        os.makedirs(os.path.join(project_dir,str(index)))
    file_path = os.path.join(project_dir,str(index), file_name)
    with open(file_path, 'w') as f:
        f.write(func)

if __name__ == '__main__':


    select_file_id = ['186432', '186610', '183889', '181346']
    c_root = '/scratch/c00590656/vulnerability/LineVul/data/big-vul_dataset'
    c_output = '/scratch/c00590656/vulnerability/DeepWukong/data/msr/source-code/'

    flaw_line_index_dict = {}
    for file in ['val.csv','test.csv', 'train.csv']: # , 'test.csv', 'train.csv'
        file_path = os.path.join(c_root, file)
        file_df = pd.read_csv(file_path)
        print(file_df.columns.values)
        for ii, row in file_df.iterrows():
            func_before = row['func_before']
            func_after = row['func_after']
            CVE_ID = row['CVE ID']
            CWE_ID =  row['CWE ID']
            target = row['target']
            index =  row['index']
            project = row['project'].strip().lower()
            flaw_line_index = row['flaw_line_index']
            if target == 1:
                print("index:", index)
                print("flaw_line_index:", flaw_line_index)

                if pd.isna(flaw_line_index):
                    flaw_line_index = []
                elif isinstance(flaw_line_index, str) and "," in flaw_line_index:
                    flaw_line_index = [int(item) for item in flaw_line_index.split(',')]
                else:
                    flaw_line_index = [int(flaw_line_index)]

                print("flaw_line_index (parsed):", flaw_line_index)

                if flaw_line_index != []:
                    flaw_line_index_max = max(flaw_line_index)

                    flaw_line_index_dict[index] = {"index":index,
                                                   "flaw_line_index":flaw_line_index,
                                                   "flaw_line_index_max":flaw_line_index_max,
                                                   "target":target,
                                                   "project":project,
                                                   "func_before":func_before}



            # if str(index) in select_file_id:
            if True:
                # print(file_path)
                # print(func_before)
                if len(project)==0:
                    project = 'unknown'
                project_dir = os.path.join(c_output, file[:-4], project)
                if not os.path.exists(project_dir):
                    os.makedirs(project_dir)

                file_name = str(index)
                if target == 0:
                    file_name_nv = file_name + '_func_before_target_0.c'
                    write_c(project_dir, file_name_nv, str(index), func_before)
                elif target == 1:
                    file_name_nv = file_name + '_func_after_target_0.c'
                    file_name_v = file_name + '_func_before_target_1.c'

                    write_c(project_dir, file_name_nv, str(index), func_after)
                    write_c(project_dir, file_name_v, str(index) , func_before)