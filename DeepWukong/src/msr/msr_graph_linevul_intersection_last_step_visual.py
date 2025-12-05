import pickle

import matplotlib.pyplot as plt
import os
import pandas as pd
import networkx as nx
from tqdm import tqdm



def expand_ranges(ranges):
    """Expand list of {'start_offset': int, 'end_offset': int} to a set of indices."""
    indices = set()
    for r in ranges:
        indices.update(range(r['start_offset'], r['end_offset']))
    return indices

def draw_code_with_underlines_scaled(text, green_ranges, black_ranges, full_path, flaw_line_index_dict, file_id):
    flaw_line_index_max = flaw_line_index_dict[int(file_id)]['flaw_line_index_max']
    flaw_line_index = flaw_line_index_dict[int(file_id)]['flaw_line_index']
    green_indices = expand_ranges(green_ranges)
    black_indices = expand_ranges(black_ranges)
    print("\nfile_id:", file_id)
    print("black_indices:", black_indices)
    print("flaw_line_index:", flaw_line_index)
    print("flaw_line_index_max:", flaw_line_index_max)


    lines = text.split('\n')
    y = 1.0
    char_index = 0
    # 计算每一行的起始char_index
    line_start_indices = []
    idx = 0
    for line in lines:
        line_start_indices.append(idx)
        idx += len(line) + 1  # +1 for '\n'

    # 获取 flaw_line_index 所在行的字符索引范围
    flaw_line_char_indices = set()
    for line_no in flaw_line_index:
        if 0 <= line_no < len(line_start_indices):
            start = line_start_indices[line_no]
            end = start + len(lines[line_no])
            flaw_line_char_indices.update(range(start, end))

    if flaw_line_char_indices:
        flaw_line_char_indices_max = max(flaw_line_char_indices)
        print("flaw_line_char_indices_max:", flaw_line_char_indices_max)
        black_indices_black = {i for i in black_indices if i <= flaw_line_char_indices_max}
        black_indices_yellow = {i for i in black_indices if i > flaw_line_char_indices_max}
        print("black_indices_black:", black_indices_black)
        print("black_indices_yellow:", black_indices_yellow)
    else:
        print("flaw_line_char_indices is empty")
        black_indices_black = flaw_line_char_indices
        black_indices_yellow = {}




    fig, ax = plt.subplots(figsize=(16, 20))  # Increased figure size
    ax.axis('off')

    # Further reduced sizes for tight layout
    line_spacing = 0.03
    char_spacing = 0.01
    char_width = 0.01
    font_size = 12
    underline_offset = line_spacing * 0.3  # 红线在下
    doubleline_gap = line_spacing * 0.2  # 黑线再下方一点

    for line in lines:
        x = 0.0
        for char in line:
            ax.text(x, y, char, fontsize=font_size, va='center', family='monospace')

            offset_idx = 0  # 用于错开多条下划线的距离

            if char_index in flaw_line_char_indices:
                y_offset = -underline_offset - offset_idx * 0.005
                ax.plot([x, x + char_width], [y + y_offset, y + y_offset], color='red', linewidth=1)
                offset_idx += 1

            if char_index in green_indices:
                y_offset = -underline_offset - offset_idx * 0.005
                ax.plot([x, x + char_width], [y + y_offset, y + y_offset], color='green', linewidth=1)
                offset_idx += 1

            if char_index in black_indices_black:
                y_offset = -underline_offset - offset_idx * 0.005
                ax.plot([x, x + char_width], [y + y_offset, y + y_offset], color='black', linewidth=1)
                offset_idx += 1

            if char_index in black_indices_yellow:
                y_offset = -underline_offset - offset_idx * 0.005
                ax.plot([x, x + char_width], [y + y_offset, y + y_offset], color='gold', linewidth=1)
                offset_idx += 1

            x += char_spacing
            char_index += 1

        char_index += 1  # for '\n'
        y -= line_spacing

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    plt.tight_layout()


    full_path_jpg = full_path.replace("XFG", "XFG_Intersection_Visual") + ".jpg"
    full_path_pkl = full_path.replace("XFG", "XFG_Intersection_Visual") + ".pkl"
    full_path_json = full_path.replace("XFG", "XFG_Intersection_Visual") + ".json"
    # 只取目录部分（不包含文件名）
    dir_path = os.path.dirname(full_path_jpg)
    # 创建目录（如果已存在不会报错）
    os.makedirs(dir_path, exist_ok=True)

    print("full_path_jpg:", full_path_jpg)
    plt.savefig(full_path_jpg)
    intersection_res = {"text":text, "green_ranges":green_ranges, "black_ranges":black_ranges}
    with open(full_path_pkl, "wb") as f:
        pickle.dump(intersection_res, f)

    with open(full_path_json, "w") as f:
        json.dump(intersection_res, f)

    plt.show()







def rank_for_linevul_tokens(file_id_list, linevul_token_scores_dir):



    top_k_tokens = []
    linevul_token_scores_dict = {}
    for file_id in file_id_list:
        file_path = os.path.join(linevul_token_scores_dir, file_id)
        df = pd.read_excel(file_path)
        # print(df)
        one_file_token_information_list = []
        for index, row in df.iterrows():
            token = row['token']
            weight = row['weight']
            one_file_token_information_list.append({'token': token, 'weight': weight, "start_offset": row['start_offset'],
                                     "end_offset": row['end_offset']})
        one_file_token_information_list = sort_tokens_and_get_rank_id(one_file_token_information_list)
        linevul_token_scores_dict[file_id] = one_file_token_information_list
    return linevul_token_scores_dict


def sort_tokens_and_get_rank_id(token_data):
    from pprint import pprint

    # # 你的数据
    # token_data = [
    #     {'token': 'page', 'start_offset': 68, 'end_offset': 72, 'weight': 0.007654893212020397},
    #     {'token': '_', 'start_offset': 72, 'end_offset': 73, 'weight': 0.01025454793125391},
    #     {'token': 'index', 'start_offset': 73, 'end_offset': 78, 'weight': -0.003516454914850848},
    #     # ...（此处省略部分数据，为演示而截取前几个）
    # ]

    # 实际上应该是你粘贴的整个大列表
    # 可以把 token_data 替换成你的完整数据

    # 排序，记录原始顺序
    for i, item in enumerate(token_data):
        item["original_order"] = i

    # 按 weight 降序、然后按原始顺序升序排序
    sorted_tokens = sorted(token_data, key=lambda x: (-x["weight"], x["original_order"]))

    # 加上 rank，从1开始
    for rank, item in enumerate(sorted_tokens, start=1):
        item["rank_id"] = rank

    # 移除 original_order，如果你不需要保留
    for item in sorted_tokens:
        item.pop("original_order")

    sorted_tokens = [item for item in sorted_tokens if item['rank_id'] <= 50]

    return sorted_tokens

def get_label_positive_predict_positive_id():
    linevul_token_scores_dir = '/scratch/c00590656/vulnerability/LineVul/linevul/shap/saved_models_MSR/token_weight_map'
    linevul_token_scores_dir = '/scratch/c00590656/vulnerability/LineVul/linevul/shap/saved_models_MSR/token_weight_map_predict_positive_label_positive'
    file_id_list_original = os.listdir(linevul_token_scores_dir)
    file_id_list = [item.split('.')[0] for item in file_id_list_original ]
    linevul_token_scores_dict = rank_for_linevul_tokens(file_id_list_original, linevul_token_scores_dir)

    return file_id_list, linevul_token_scores_dict


def read_xfg_label(full_path):
    pkl_file = full_path.replace("XFG_LineVul_Map", "XFG").replace(".linevul_map.json", "")


    if os.path.getsize(pkl_file) == 0:
        print("\n\n\n**********error pkl_file:", pkl_file)
        return 2   # 2 meaning error

    one_res = {}
    xfg = nx.read_gpickle(pkl_file)
    label_slicing = xfg.graph['label']
    num_nodes_slicing = len(xfg.nodes)
    num_nodes_func = 0

    one_res['xfg_file'] = pkl_file
    one_res['label_slicing'] = label_slicing
    one_res['num_nodes_slicing'] = num_nodes_slicing

    return label_slicing


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



def sort_tokens_and_get_rank_id(token_data):
    from pprint import pprint

    # # 你的数据
    # token_data = [
    #     {'token': 'page', 'start_offset': 68, 'end_offset': 72, 'weight': 0.007654893212020397},
    #     {'token': '_', 'start_offset': 72, 'end_offset': 73, 'weight': 0.01025454793125391},
    #     {'token': 'index', 'start_offset': 73, 'end_offset': 78, 'weight': -0.003516454914850848},
    #     # ...（此处省略部分数据，为演示而截取前几个）
    # ]

    # 实际上应该是你粘贴的整个大列表
    # 可以把 token_data 替换成你的完整数据

    # 排序，记录原始顺序
    for i, item in enumerate(token_data):
        item["original_order"] = i

    # 按 weight 降序、然后按原始顺序升序排序
    sorted_tokens = sorted(token_data, key=lambda x: (-x["weight"], x["original_order"]))

    # 加上 rank，从1开始
    for rank, item in enumerate(sorted_tokens, start=1):
        item["rank_id"] = rank

    # 移除 original_order，如果你不需要保留
    for item in sorted_tokens:
        item.pop("original_order")

    sorted_tokens = [item for item in sorted_tokens if item['rank_id'] <= 100]
    return sorted_tokens


def read_flaw_lines_index():
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

                if pd.isna(flaw_line_index):
                    flaw_line_index = []
                elif isinstance(flaw_line_index, str) and "," in flaw_line_index:
                    flaw_line_index = [int(item) for item in flaw_line_index.split(',')]
                else:
                    flaw_line_index = [int(flaw_line_index)]


                if flaw_line_index != []:
                    flaw_line_index_max = max(flaw_line_index)

                    flaw_line_index_dict[index] = {"index":index,
                                                   "flaw_line_index":flaw_line_index,
                                                   "flaw_line_index_max":flaw_line_index_max,
                                                   "target":target,
                                                   "project":project,
                                                   "func_before":func_before}

    return flaw_line_index_dict



if __name__ == '__main__':

    root = '/scratch/c00590656/vulnerability/DeepWukong/data/msr/source-code/'
    all_c_files, all_c_files_dict = get_all_c_files(root)
    flaw_line_index_dict = read_flaw_lines_index()

    file_id_list, linevul_token_scores_dict = get_label_positive_predict_positive_id()
    c_root = '/scratch/c00590656/vulnerability/DeepWukong/data/msr/XFG_LineVul_Map'
    output_dir = '/scratch/c00590656/vulnerability/DeepWukong/src/msr/msr_graph_linevul_intersection_last_step_analysis_res'


    root_dir_list = os.listdir(c_root)
    # exit()
    ii = 0
    top_k_tokens = []
    for file_id in tqdm(root_dir_list, desc="Processing files"):

        if file_id in file_id_list:
            pass
        else:
            continue

        if file_id in ['179655', '180688', '181013', '183528', '183712', '184827', '185116', '185352', '186157', '186623', '186678', '187878', '187989']:
        # if file_id in ['179655']:
            pass
        else:
            continue

        print("\n\nfile_id:", file_id)

        ii = ii + 1

        c_file_path = all_c_files_dict[file_id]

        with open(c_file_path, "r") as f:
            original_code = f.read()



        file_id_dir = os.path.join(c_root, file_id)
        one_file_id_files = []
        all_xfg_res = []

        for dirpath, dirnames, filenames in os.walk(file_id_dir):
            for filename in filenames:
                full_path = os.path.join(dirpath, filename)
                one_file_id_files.append(full_path)
                label_slicing = read_xfg_label(full_path)
                # print("label_slicing:", label_slicing)
                if label_slicing == 1 or (label_slicing == 0 and 'vulnerable' in full_path):
                    pass
                else:
                    continue

                # print("full_path:", full_path)
                # print("label_slicing:", label_slicing)
                import ast
                import json


                with open(full_path, 'r') as f:
                    content = f.read()
                content = content.replace("nan", "None")
                # 使用 ast.literal_eval 尝试安全地解析成 Python 字典
                one_slicing_res = ast.literal_eval(content)
                # print(one_slicing_res)

                # node_mapped_linevul_tokens_concat = []
                # for node_index, node_corr_information in one_slicing_res.items():
                #     if node_index == 1:
                #         continue
                #     node_mapped_linevul_tokens = node_corr_information['node_mapped_linevul_tokens']
                #     node_mapped_linevul_tokens_concat += node_mapped_linevul_tokens

                node_mapped_linevul_tokens_concat = []
                for node_index, node_corr_information in one_slicing_res.items():
                    if node_index == 1:
                        continue
                    code_updated = node_corr_information['code_updated']
                    location_updated = node_corr_information['location_updated']
                    if "#" in location_updated:
                        print("location_updated:", location_updated)
                        start_offset, end_offset = location_updated.split("#")
                        token = code_updated
                        wight = 1
                        one_node = {"start_offset":int(start_offset),
                                    "end_offset":int(end_offset),
                                    "token":token,
                                    "wight":wight,}
                        node_mapped_linevul_tokens_concat.append(one_node)



                green_ranges = linevul_token_scores_dict[file_id+".pkl.xlsx"]
                black_ranges = node_mapped_linevul_tokens_concat
                # print("original_code:\n", original_code)
                # print("green_ranges:\n", green_ranges)
                # print("black_ranges:\n", black_ranges)
                draw_code_with_underlines_scaled(original_code, green_ranges, black_ranges, full_path, flaw_line_index_dict, file_id)






