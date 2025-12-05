import os
import  pandas as pd
import os.path
from os.path import join, exists
from typing import List, Set, Tuple, Dict
import pandas as pd
import os
import pandas as pd
import os
import networkx as nx
import matplotlib.pyplot as plt
from sympy.physics.mechanics import potential_energy
from tqdm import tqdm
import json
import glob
import pandas as pd
import re
# from  msr_graph_linevul_intersection import update_joern_nodes_location
def read_csv(csv_file_path: str) -> List:
    """
    read csv file
    """
    assert exists(csv_file_path), f"no {csv_file_path}"
    data = []
    with open(csv_file_path) as fp:
        header = fp.readline()
        header = header.strip()
        h_parts = [hp.strip() for hp in header.split('\t')]
        for line in fp:
            line = line.strip()
            instance = {}
            lparts = line.split('\t')
            for i, hp in enumerate(h_parts):
                if i < len(lparts):
                    content = lparts[i].strip()
                else:
                    content = ''
                instance[hp] = content
            data.append(instance)
        return data


if __name__ == '__main__':

    # nodes_csv = '/scratch/c00590656/vulnerability/DeepWukong/data/msr/csv/scratch/c00590656/vulnerability/DeepWukong/data/msr/source-code/test/chrome/183528/183528_func_before_target_1.c/nodes.csv'
    #
    # nodes_df = read_csv(nodes_csv)
    # nodes_df = pd.DataFrame(nodes_df)
    # nodes_df.to_excel(nodes_csv+".xlsx", index=False)
    # exit()


    xfg_already = '/scratch/c00590656/vulnerability/DeepWukong/src/data/msr/XFG'
    xfg_already_file_id = os.listdir(xfg_already)


    already_id = '/scratch/c00590656/vulnerability/DeepWukong/data/msr/XFG_Intersection_Visual_LineVul_Map'
    already_id = '/scratch/c00590656/vulnerability/DeepWukong/src/data/msr/XFG_Intersection_Visual_LineVul_Map'
    already_id = os.listdir(already_id)
    # already_id = ['179655', '180688', '181013', '183528', '183712', '184827', '185116', '185352', '186157', '186623', '186678', '187878', '187989']
    output_dir = '/scratch/c00590656/vulnerability/DeepWukong/src/data/msr/XFG_Intersection_Visual_LineVul_Map'

    c_root = '/scratch/c00590656/vulnerability/LineVul/data/big-vul_dataset'
    files_list = ['test.csv', 'val.csv', 'train.csv']

    for file in files_list:
        file_path = os.path.join(c_root, file)
        file_df = pd.read_csv(file_path)
        print(file_df.columns.values)
        unique_values = file_df['project'].unique()
        print("unique_values:", unique_values)
        # C_Project = ['openssl', 'qemu','linux-core', 'libxml2', 'libarchive', 'libgit2', 'tcpdump']
        # file_df_filter = file_df[file_df['project'].isin(C_Project)]
        #
        # for ii, row in file_df_filter.iterrows():
        #     index = row['index']
        #     if str(index) not in xfg_already_file_id:
        #         if row['target'] == 1:
        #             print(index, "   ", row['project'], '  ', row['target'])
        #
        # exit()

        output_dir = '/scratch/c00590656/vulnerability/LineVul/data/big-vul_dataset/example'
        file_df_selected = file_df[file_df['index'].isin(['181024', '179655', 181024, 179655])]
        for ii, row in file_df_selected.iterrows():
            index = row['index']

            func_before = row['func_before']
            func_after = row['func_after']
            row_information = row.to_json(orient='records')

            index_dir = os.path.join(output_dir, str(index))
            os.makedirs(index_dir, exist_ok=True)
            with open(os.path.join(index_dir, 'func_before.txt'), 'w') as f:
                f.write(func_before)
            with open(os.path.join(index_dir, 'func_after.txt'), 'w') as f:
                f.write(func_after)
            with open(os.path.join(index_dir, 'row_information.json'), 'w') as f:
                json.dump(row_information, f)







        # for ii, row in file_df.iterrows():
        #     index = row['index']
        #     if str(index) in already_id:
        #         print("\n\n******************\nindex:\n", row['index'])
        #         print("\ntarget:\n", row['target'])
        #         print("\ncommit_id:\n", row['commit_id'])
        #         print("\nfunc_before:\n", row['func_before'])
        #         print("\nflaw_line_index:\n", row['flaw_line_index'])
        #         print("\nflaw_line:\n", row['flaw_line'])
        #         print(row)
        #         try:
        #             output_file = os.path.join(output_dir, str(index), str(row['index'])+"_basic_information.txt")
        #             with open(output_file, 'w', encoding='utf-8') as f:
        #                 f.write(row['func_before'])
        #                 f.write("\n\ncommit_id:\n"+ row['commit_id'])
        #                 f.write("\n\nflaw_line_index:\n" + row['flaw_line_index'])
        #                 f.write("\n\nflaw_line:\n" + row['flaw_line'])
        #                 f.write("\n\ntarget:\n" + str(row['target']))
        #                 f.write("\n\nproject:\n" + row['project'])
        #                 f.write("\n\ncodeLink:\n" + row['codeLink'])
        #         except Exception as e:
        #             print(e)
        #

