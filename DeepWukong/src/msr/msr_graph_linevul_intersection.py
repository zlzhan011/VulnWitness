


import os
import pandas as pd
import os
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import glob
import pandas as pd
import re

from os.path import join, exists
from typing import List, Set, Tuple, Dict
from nodes_csv_process import  read_csv, collect_csv_files
from linevul_tokens_score import read_linevul_token_scores
from nodes_csv_process import update_joern_nodes_location
from msr_graph_linevul_intersection_last_step import msr_graph_linevul_intersection_last_step



def process_xfg_files(input_dir):
    """
    处理目录下所有 .pkl 文件，生成可视化图像和节点属性信息

    参数:
    input_dir -- 包含 .pkl 文件的目录
    output_dir -- 输出图像和属性信息的目录
    max_files -- 最大处理文件数，None表示处理所有文件
    """

    directory = '/scratch/c00590656/vulnerability/DeepWukong/data/msr/csv/scratch/c00590656/vulnerability/DeepWukong/data/msr/source-code'
    csv_files, csv_dict, v_instance_dict= collect_csv_files(directory)



    # 递归查找所有 .pkl 文件
    pkl_files = []
    ii = 0
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.pkl'):
                pkl_files.append(os.path.join(root, file))
                ii += 1
                print("ii:", ii)
                if ii >= 100:
                    break

        if ii >= 100:
            break


    # print(pkl_files)

    final_output = []

    for pkl_file in tqdm(pkl_files):
        print("\n\n\n**********pkl_file:", pkl_file)
        one_res = {}
        xfg = nx.read_gpickle(pkl_file)
        label_slicing = xfg.graph['label']
        num_nodes_slicing = len(xfg.nodes)
        num_nodes_func = 0

        one_res['xfg_file'] =pkl_file
        one_res['label_slicing'] = label_slicing
        one_res['num_nodes_slicing'] = num_nodes_slicing


        # 创建输出文件路径
        rel_path = os.path.relpath(pkl_file, input_dir)
        base_name = os.path.splitext(rel_path)[0]
        index_name = rel_path.split('/')[0]

        # if index_name != '169996':
        #     continue
        # nodes_csv = ""
        # for one_csv_name in csv_files:
        #     if str(index_name) in one_csv_name:
        #         if 'node' in one_csv_name:
        #             nodes_csv = one_csv_name
        #             break

        nodes_csv = ""
        if str(index_name) in csv_dict:
            nodes_csv_potential = csv_dict[str(index_name)]
            for item in nodes_csv_potential:
                if 'target' in item and '.c' in item and 'node' in item:
                    nodes_csv += item
                    break


        print("\nnodes_csv:", nodes_csv)

        if len(nodes_csv)<=2:
            print("chect the path of nodes csv")
        else:
            nodes_df = read_csv(nodes_csv)

            nodes_df_dict = {}
            ii = 0
            for row in nodes_df:
                row_code = row['code']
                nodes_df_dict[ii] = row_code
                ii += 1

            num_nodes_func = ii
            one_res['num_nodes_func'] = num_nodes_func
            one_res['nodes_csv'] = nodes_csv
            print("one_res:\n", one_res)

        final_output.append(one_res)


    final_output = pd.DataFrame(final_output)
    final_output.to_csv('/scratch/c00590656/vulnerability/DeepWukong/data/msr/XFG_analysis/node_proportion/node_proportion_test.csv')







def process_xfg_linevul_node_map(input_dir):

    """
    处理目录下所有 .pkl 文件，生成可视化图像和节点属性信息

    参数:
    input_dir -- 包含 .pkl 文件的目录
    output_dir -- 输出图像和属性信息的目录
    max_files -- 最大处理文件数，None表示处理所有文件
    """

    # linevul_token_scores_dir = '/work/lzhan011/vulnerability/LineVul/linevul/shap/saved_models_MSR/token_weight_map' # loni
    linevul_token_scores_dir = '/scratch/c00590656/vulnerability/LineVul/linevul/shap/saved_models_MSR/token_weight_map'

    file_id_list = os.listdir(linevul_token_scores_dir)
    file_id_list = [item.split('.')[0] for item in file_id_list ]


    # read nodes.csv
    directory = '/scratch/c00590656/vulnerability/DeepWukong/data/msr/csv/scratch/c00590656/vulnerability/DeepWukong/data/msr/source-code'
    # directory = os.path.join(directory, 'test')
    csv_files, csv_dict, v_instance_dict= collect_csv_files(directory)

    csv_dict_keys = set(list(csv_dict.keys()))
    csv_dict_keys = set(file_id_list)



    # read graph     pkl file
    # 递归查找所有 .pkl 文件
    pkl_files = []
    ii = 0
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.pkl'):
                file_id = file.split('.')[0]
                if file_id in csv_dict_keys:
                    pkl_files.append(os.path.join(root, file))
                    ii += 1
                    print("ii:", ii)
                    if ii >= 100:
                        break

        if ii >= 100:
            break


    # print(pkl_files)

    final_output = []

    for pkl_file in tqdm(pkl_files):
        print("\n\n\n**********pkl_file:", pkl_file)
        one_res = {}
        xfg = nx.read_gpickle(pkl_file)
        label_slicing = xfg.graph['label']
        num_nodes_slicing = len(xfg.nodes)
        num_nodes_func = 0

        one_res['xfg_file'] =pkl_file
        one_res['label_slicing'] = label_slicing
        one_res['num_nodes_slicing'] = num_nodes_slicing


        # 创建输出文件路径
        rel_path = os.path.relpath(pkl_file, input_dir)
        base_name = os.path.splitext(rel_path)[0]
        index_name = rel_path.split('/')[0]
        index_linevul_token_scores = read_linevul_token_scores(index_name, linevul_token_scores_dir)
        print("index_linevul_token_scores:\n", index_linevul_token_scores)
        if type(index_linevul_token_scores) == list:
            continue

        # if index_name != '169996':
        #     continue
        # nodes_csv = ""
        # for one_csv_name in csv_files:
        #     if str(index_name) in one_csv_name:
        #         if 'node' in one_csv_name:
        #             nodes_csv = one_csv_name
        #             break

        nodes_csv = ""
        if str(index_name) in csv_dict:
            nodes_csv_potential = csv_dict[str(index_name)]
            for item in nodes_csv_potential:
                if 'target' in item and '.c' in item and 'node' in item:
                    nodes_csv += item
                    break


        print("\nnodes_csv:", nodes_csv)

        if len(nodes_csv)<=2:
            print("chect the path of nodes csv")
        else:
            nodes_df = read_csv(nodes_csv)

            nodes_df_dict = {}
            ii = 0
            for row in nodes_df:
                row_code = row['code']
                nodes_df_dict[ii] = row_code
                ii += 1

            num_nodes_func = ii
            one_res['num_nodes_func'] = num_nodes_func
            one_res['nodes_csv'] = nodes_csv
            print("one_res:\n", one_res)

        final_output.append(one_res)


    final_output = pd.DataFrame(final_output)
    # final_output.to_csv('/scratch/c00590656/vulnerability/DeepWukong/data/msr/XFG_analysis/node_proportion/node_proportion_test.csv')






import os

def get_file_id_graph_map(input_dir):
    file_id_graph_map = {}
    duplicate_count = 0

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.pkl'):
                full_path = os.path.join(root, file)
                path_parts = root.strip(os.sep).split(os.sep)
                XFG_index = path_parts.index("XFG")
                file_id_graph = path_parts[XFG_index+1]
                if file_id_graph in file_id_graph_map:
                    file_id_graph_map[file_id_graph].append(full_path)
                    duplicate_count += 1
                else:
                    file_id_graph_map[file_id_graph] = [full_path]

    print(f"Found {duplicate_count} duplicate file_id entries.")
    # print("file_id_graph_map:", file_id_graph_map)
    return file_id_graph_map





def get_joern_nodes(index_name, csv_dict):
    nodes_csv = ""
    if str(index_name) in csv_dict:
        nodes_csv_potential = csv_dict[str(index_name)]
        for item in nodes_csv_potential:
            if ('target' in item) and ('.c' in item) and ('node' in item) and ('func_before' in item):
                nodes_csv += item
                break

    print("\nnodes_csv:", nodes_csv)
    nodes_df = []
    if len(nodes_csv) <= 2:
        print("chect the path of nodes csv")
    else:
        nodes_df = read_csv(nodes_csv)

    return nodes_df, nodes_csv






def joern_nodes_and_slicing_graph_intersection(joer_nodes, xfg_nodes):

    xfg_nodes_add_code_information = {}
    for xfg_nodes_index in xfg_nodes:
        xfg_nodes_item = joer_nodes[xfg_nodes_index]
        xfg_nodes_add_code_information[xfg_nodes_index] = xfg_nodes_item

    return xfg_nodes_add_code_information






# 定义偏移提取函数
def extract_by_offset(filename, start_line, start_column, end_offset):
    with open(filename, "r") as f:
        full_text = f.read()

    lines = full_text.splitlines(keepends=True)
    start_index = sum(len(lines[i]) for i in range(start_line - 1)) + start_column
    return full_text, full_text[start_index:end_offset]



import re

def get_child_location_from_parent(filename, parent_location, parent_code, child_code):
    # location 形如 '2:2:60:114' -> line 2, column 2, char_offset 60, end_offset 114
    start_line, start_col, start_offset, _ = map(int, parent_location.split(":"))

    # 读取整个文件内容
    with open(filename, "r") as f:
        full_text = f.read()

    # 在 parent_code 中找 child_code 的相对位置
    rel_index = parent_code.find(child_code)
    if rel_index == -1:
        raise ValueError("Child code not found in parent code.")

    # 子 code 在原始文件中的实际 offset
    abs_offset = start_offset + rel_index

    # 找出该 offset 对应的行号和列号
    current_offset = 0
    for line_number, line in enumerate(full_text.splitlines(keepends=True), start=1):
        next_offset = current_offset + len(line)
        if current_offset <= abs_offset < next_offset:
            char_index = abs_offset - current_offset
            return {
                "line": line_number,
                "column": char_index,
                "abs_offset": abs_offset
            }
        current_offset = next_offset

    raise ValueError("Offset exceeds file length.")



def get_all_c_files(root_dir):
    all_files = []
    all_files_dict = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            file_id = f.split('_')[0]
            full_path = os.path.join(dirpath, f)
            if 'func_before' in full_path:
                all_files.append(full_path)
                all_files_dict[file_id] = full_path
    return all_files, all_files_dict


#
# def update_joern_nodes_location_bak(joern_nodes, c_file_path):
#     joern_nodes_updated = []
#     for i in range(len(joern_nodes)):
#         print("i:", i)
#         one_node = joern_nodes[i]
#         print("one_node---:   ", one_node)
#         one_node_location = one_node['location']
#         one_node_code = one_node['code']
#         one_node_type = one_node['type']
#         if i == 0:
#             one_node['location_updated'] = ""
#             one_node['code_updated'] = ""
#         else:
#             if len(one_node_code.strip()) >= 0 and one_node_type != 'Symbol':
#                 if ":" in one_node_location:
#                     line, col, loc_3, end = one_node_location.split(":")
#                     parent_location = one_node_location
#                     parent_code = one_node['code']
#                     parent_type = one_node['type']
#                     original_code, parent_code_original = extract_by_offset(c_file_path, int(line), int(col), int(end))
#                     node_c_code_original_start_index = original_code.index(parent_code_original)
#                     node_c_code_original_end_index = node_c_code_original_start_index + len(parent_code_original)
#                     parent_location_updated = str(node_c_code_original_start_index) + "#" + str(node_c_code_original_end_index)
#                     one_node['location_updated'] = parent_location_updated
#                     one_node['code_updated'] = parent_code_original
#                 else:
#                     child_code = one_node_code
#                     line, col, loc_3, end = parent_location.split(":")
#
#                     child_type = one_node['type']
#                     if parent_type =='Function' and child_type == 'FunctionDef':
#                         child_code_tmp = child_code
#                         child_code = parent_code
#                         parent_code = child_code_tmp
#
#
#                     if child_code in parent_code:
#
#                         child_location = get_child_location_from_parent(c_file_path, parent_location, parent_code, child_code)
#                         line, col,  end = child_location['line'], child_location['column'], child_location['abs_offset']
#                         original_code, child_code_original = extract_by_offset(c_file_path, int(line), int(col), int(end))
#
#                         relative_location_start = parent_code_original.index(child_code_original)
#                         relative_location_end = relative_location_start + len(child_code_original)
#
#                         parent_location_updated_start = parent_location_updated.split("#")[0]
#                         child_location_start = int(parent_location_updated_start) + relative_location_start
#                         child_location_end = child_location_start + len(child_code_original)
#
#                         child_location_updated = str(child_location_start) + "#" + str(child_location_end)
#                         one_node['location_updated'] = child_location_updated
#                         one_node['code_updated'] = child_code_original
#                     else:
#                         one_node['location_updated'] = ""
#                         one_node['code_updated'] = ""
#
#
#
#         joern_nodes_updated.append(one_node)
#
#
#     return joern_nodes_updated







def process_xfg_linevul_node_map_v2(input_dir):

    """
    处理目录下所有 .pkl 文件，生成可视化图像和节点属性信息

    参数:
    input_dir -- 包含 .pkl 文件的目录
    output_dir -- 输出图像和属性信息的目录
    max_files -- 最大处理文件数，None表示处理所有文件
    """

    root = '/scratch/c00590656/vulnerability/DeepWukong/data/msr/source-code/'
    all_c_files, all_c_files_dict = get_all_c_files(root)

    # read nodes.csv
    directory = '/scratch/c00590656/vulnerability/DeepWukong/data/msr/csv/scratch/c00590656/vulnerability/DeepWukong/data/msr/source-code'
    # directory = os.path.join(directory, 'test')
    csv_files, csv_dict, v_instance_dict= collect_csv_files(directory)



    file_id_graph_map = get_file_id_graph_map(input_dir)
    # linevul_token_scores_dir = '/work/lzhan011/vulnerability/LineVul/linevul/shap/saved_models_MSR/token_weight_map' # loni
    linevul_token_scores_dir = '/scratch/c00590656/vulnerability/LineVul/linevul/shap/saved_models_MSR/token_weight_map'
    linevul_token_scores_dir = '/scratch/c00590656/vulnerability/LineVul/linevul/shap/saved_models_MSR/token_weight_map_predict_positive_label_positive'
    file_id_list = os.listdir(linevul_token_scores_dir)
    file_id_list = [item.split('.')[0] for item in file_id_list ]


    nn = 0
    skip_num = 0

    map_finished = os.listdir('/scratch/c00590656/vulnerability/DeepWukong/data/msr/XFG_LineVul_Map')

    # file_id_list = ['179655', '180688', '181013', '183528', '183712', '184827', '185116', '185352', '186157', '186623', '186678', '187878', '187989']
    # file_id_list = ['178394', '183199','178134',  '183024', '180175', '178422', '178638', '178360', '181024' , '182416', '183206', '178100', '179853','178356', '178640', '178184', '181067', '181054', '181950', '178625', '178105', '181701', '181132', '181028', '181107', '178458']
    for file_id_linevul in file_id_list:
        nn += 1
        print("file_id_linevul: ", file_id_linevul, 'nn: ', nn)

        # if str(file_id_linevul) in  ['179655', '180688', '181013', '183528', '183712', '184827', '185116', '185352', '186157', '186623', '186678', '187878', '187989']:
        #     pass
        # else:
        #     continue


        # if str(file_id_linevul) in map_finished:
        #     continue


        index_linevul_token_scores = read_linevul_token_scores(file_id_linevul, linevul_token_scores_dir)
        if isinstance(index_linevul_token_scores, list):
            continue
        else:
            index_linevul_token_scores.to_excel(os.path.join('/scratch/c00590656/vulnerability/DeepWukong/data/msr/linevul_graph_intersection/', str(file_id_linevul)+"_linevul_token_weight.xlsx"))

        if file_id_linevul not in file_id_graph_map:
            skip_num += 1
            print("skip number: ", skip_num)
            continue

        graph_pkl_files = file_id_graph_map[file_id_linevul]

        joern_nodes, joern_nodes_csv = get_joern_nodes(file_id_linevul, csv_dict)
        c_file_path = all_c_files_dict[file_id_linevul]
        print("c_file_path:", c_file_path)
        joern_nodes = update_joern_nodes_location(joern_nodes, c_file_path)

        final_output = []

        for pkl_file in tqdm(graph_pkl_files):
            print("\n\n\n**********pkl_file:", pkl_file)

            if os.path.getsize(pkl_file) == 0:
                print("\n\n\n**********error pkl_file:", pkl_file)
                continue

            one_res = {}
            xfg = nx.read_gpickle(pkl_file)
            label_slicing = xfg.graph['label']
            num_nodes_slicing = len(xfg.nodes)
            num_nodes_func = 0

            one_res['xfg_file'] = pkl_file
            one_res['label_slicing'] = label_slicing
            one_res['num_nodes_slicing'] = num_nodes_slicing

            xfg_nodes = list(xfg.nodes)
            xfg_nodes_add_code_information = joern_nodes_and_slicing_graph_intersection(joern_nodes, xfg_nodes)


            # 写入文件
            with open(os.path.join('/scratch/c00590656/vulnerability/DeepWukong/data/msr/linevul_graph_intersection/',
                             str(file_id_linevul) + "_xfg_nodes_add_code_information.json"), 'w') as f:
                json.dump(xfg_nodes_add_code_information, f)

            msr_graph_linevul_intersection_last_step(index_linevul_token_scores, xfg_nodes_add_code_information, pkl_file)




if __name__ == '__main__':
    c_root = '/scratch/c00590656/vulnerability/DeepWukong/data/msr/XFG'
    c_root = '/scratch/c00590656/vulnerability/DeepWukong/src/data/msr/XFG'
    # process_xfg_files(c_root)
    # process_xfg_linevul_node_map(c_root)
    process_xfg_linevul_node_map_v2(c_root)