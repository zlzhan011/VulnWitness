import pickle

import matplotlib.pyplot as plt
import os
import pandas as pd
import networkx as nx




def expand_ranges(ranges):
    """Expand list of {'start_offset': int, 'end_offset': int} to a set of indices."""
    indices = set()
    for r in ranges:
        indices.update(range(r['start_offset'], r['end_offset']))
    return indices

# def draw_code_with_underlines_scaled(text, green_ranges, black_ranges, full_path):
#     green_indices = expand_ranges(green_ranges)
#     black_indices = expand_ranges(black_ranges)
#
#     fig, ax = plt.subplots(figsize=(16, 20))  # Increased figure size
#     ax.axis('off')
#
#     # Further reduced sizes for tight layout
#     line_spacing = 0.03
#     char_spacing = 0.01
#     char_width = 0.01
#     font_size = 12
#     underline_offset = line_spacing * 0.3  # 红线在下
#     doubleline_gap = line_spacing * 0.2  # 黑线再下方一点
#
#     lines = text.split('\n')
#     y = 1.0
#     char_index = 0
#
#     for line in lines:
#         x = 0.0
#         for char in line:
#             ax.text(x, y, char, fontsize=font_size, va='center', family='monospace')
#
#             if char_index in green_indices:
#                 ax.plot([x, x + char_width], [y - underline_offset, y - underline_offset], color='red', linewidth=1)
#
#             if char_index in black_indices:
#                 offset = -underline_offset - doubleline_gap if char_index in green_indices else -underline_offset
#                 ax.plot([x, x + char_width], [y + offset, y + offset], color='black', linewidth=1)
#
#             x += char_spacing
#             char_index += 1
#
#         char_index += 1  # account for \n
#         y -= line_spacing
#
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1.1)
#     plt.tight_layout()
#
#
#     full_path_jpg = full_path.replace("XFG", "XFG_Intersection_Visual") + ".jpg"
#     full_path_pkl = full_path.replace("XFG", "XFG_Intersection_Visual") + ".pkl"
#     full_path_json = full_path.replace("XFG", "XFG_Intersection_Visual") + ".json"
#     # 只取目录部分（不包含文件名）
#     dir_path = os.path.dirname(full_path_jpg)
#     # 创建目录（如果已存在不会报错）
#     os.makedirs(dir_path, exist_ok=True)
#
#     print("full_path_jpg:", full_path_jpg)
#     plt.savefig(full_path_jpg)
#     intersection_res = {"text":text, "green_ranges":green_ranges, "black_ranges":black_ranges}
#     with open(full_path_pkl, "wb") as f:
#         pickle.dump(intersection_res, f)
#
#     with open(full_path_json, "w") as f:
#         json.dump(intersection_res, f)
#
#     plt.show()

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
                    flaw_line_index = [int(item)+1 for item in flaw_line_index.split(',')]
                else:
                    flaw_line_index = [int(flaw_line_index)+1]


                if flaw_line_index != []:
                    flaw_line_index_max = max(flaw_line_index)

                    flaw_line_index_dict[index] = {"index":index,
                                                   "flaw_line_index":flaw_line_index,
                                                   "flaw_line_index_max":flaw_line_index_max,
                                                   "target":target,
                                                   "project":project,
                                                   "func_before":func_before}

    return flaw_line_index_dict



def draw_code_with_underlines_scaled(text,
                                     green_ranges,
                                     black_line_numbers,
                                     full_path,
                                     flaw_line_index_dict, file_id,
                                     line_spacing=0.03,
                                     char_spacing=0.01,
                                     char_width=0.01,
                                     font_size=12):
    """在代码里画红色字符下划线与整行黑线并保存结果"""

    red_line_numbers = flaw_line_index_dict[int(file_id)]["flaw_line_index"]
    green_indices = expand_ranges(green_ranges)

    fig, ax = plt.subplots(figsize=(16, 20))
    ax.axis("off")

    underline_offset = line_spacing * 0.3      # 单条下划线偏移
    doubleline_gap   = line_spacing * 0.2      # 双线之间的间距

    lines   = text.split("\n")
    y       = 1.0
    char_ix = 0

    full_path_txt = full_path.replace("XFG", "XFG_Intersection_Visual") + ".txt"
    os.makedirs(os.path.dirname(full_path_txt), exist_ok=True)
    f_write = open(full_path_txt, "w")
    for line_no, line in enumerate(lines, start=1):
        x = 0.0
        for ch in line:
            ax.text(x, y, ch, fontsize=font_size,
                    va="center", family="monospace")

            if char_ix in green_indices:
                ax.hlines(y - underline_offset,
                          x, x + char_width,
                          color="green", linewidth=1)

            x += char_spacing
            char_ix += 1

        # === 整行标注：按颜色错开 ===
        if line_no in red_line_numbers:
            offset = -underline_offset - doubleline_gap
            ax.hlines(y + offset, 0.0, x,
                      color="red", linewidth=1.5,
                      zorder=6)

        if line_no in black_line_numbers:
            offset = -underline_offset - doubleline_gap - 0.01  # 再往下错开
            ax.hlines(y + offset, 0.0, x,
                      color="black", linewidth=1.5,
                      zorder=5)

            print("line_no:", line_no)
            print("line:", line)
            f_write.write("\nline_no: " + str(line_no))
            f_write.write("\nline: " + str(line))

        char_ix += 1
        y -= line_spacing

    # --------- 自动调整坐标范围，避免被裁剪 ---------
    bottom_y = y - underline_offset - doubleline_gap - 0.05   # 稍留余量
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom_y, 1.05)

    plt.tight_layout()

    # --------- 保存图片与元数据 ---------
    full_path_jpg  = full_path.replace("XFG", "XFG_Intersection_Visual") + ".jpg"
    full_path_pkl  = full_path.replace("XFG", "XFG_Intersection_Visual") + ".pkl"
    full_path_json = full_path.replace("XFG", "XFG_Intersection_Visual") + ".json"
    print("full_path_jpg:", full_path_jpg)
    os.makedirs(os.path.dirname(full_path_jpg), exist_ok=True)
    plt.savefig(full_path_jpg, dpi=300)

    meta = {"text": text,
            "green_ranges": green_ranges,
            "black_line_numbers": black_line_numbers}

    with open(full_path_pkl,  "wb") as f: pickle.dump(meta, f)
    with open(full_path_json, "w")  as f: json.dump(meta, f)

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
        return 2, ''   # 2 meaning error

    one_res = {}
    xfg = nx.read_gpickle(pkl_file)
    label_slicing = xfg.graph['label']
    num_nodes_slicing = len(xfg.nodes)
    num_nodes_func = 0

    one_res['xfg_file'] = pkl_file
    one_res['label_slicing'] = label_slicing
    one_res['num_nodes_slicing'] = num_nodes_slicing

    return label_slicing, xfg


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


def get_line_offsets(text):
    lines = text.splitlines(keepends=True)
    offsets = {}
    current_offset = 0


    for line_index, line in enumerate(lines):
        line_start = current_offset
        line_end = current_offset + len(line)
        offsets[line_index] = {"start_offset":line_start,
                        "end_offset": line_end,
                               "code":line}
        current_offset = line_end

    return offsets




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

if __name__ == '__main__':

    root = '/scratch/c00590656/vulnerability/DeepWukong/data/msr/source-code/'
    all_c_files, all_c_files_dict = get_all_c_files(root)
    flaw_line_index_dict = read_flaw_lines_index()

    file_id_list, linevul_token_scores_dict = get_label_positive_predict_positive_id()
    c_root = '/scratch/c00590656/vulnerability/DeepWukong/data/msr/XFG_LineVul_Map'
    c_root = '/scratch/c00590656/vulnerability/DeepWukong/src/data/msr/XFG_LineVul_Map'
    output_dir = '/scratch/c00590656/vulnerability/DeepWukong/src/msr/msr_graph_linevul_intersection_last_step_analysis_res'


    root_dir_list = os.listdir(c_root)
    # exit()
    ii = 0
    top_k_tokens = []
    for file_id in root_dir_list:


        if file_id in file_id_list:
            pass
        else:
            continue

        # if file_id in ['179655', '180688', '181013', '183528', '183712', '184827', '185116', '185352', '186157', '186623', '186678', '187878', '187989']:
        # if file_id in ['178394', '183712', '183199','178134',  '183024', '180175', '178422', '178638', '178360', '181024' , '182416', '183206', '178100', '179853','178356', '178640', '178184', '181067', '181054', '181950', '178625', '178105', '181701', '181132', '181028', '181107', '178458']:
        # # if file_id in ['179655']:
        #     pass
        # else:
        #     continue

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
                label_slicing, xfg = read_xfg_label(full_path)
                print("label_slicing:", label_slicing)
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
                print(one_slicing_res)



                xfg_nodes_line_number = list(xfg.nodes)
                line_number_offsets = get_line_offsets(original_code)

                xfg_nodes_line_number_map_offsets = []
                for one_line_number in xfg_nodes_line_number:
                    line_start_index = line_number_offsets[one_line_number]['start_offset']
                    line_end_index = line_number_offsets[one_line_number]['end_offset']

                    wight = 1
                    one_node = {"start_offset": int(line_start_index),
                                "end_offset": int(line_end_index),
                                "token": line_number_offsets[one_line_number]['code'],
                                "wight": wight, }
                    xfg_nodes_line_number_map_offsets.append(one_node)


                green_ranges = linevul_token_scores_dict[file_id+".pkl.xlsx"]
                black_ranges = xfg_nodes_line_number_map_offsets
                # print("original_code:\n", original_code)
                # print("green_ranges:\n", green_ranges)
                # print("black_ranges:\n", black_ranges)
                # draw_code_with_underlines_scaled(original_code, green_ranges, black_ranges, full_path)
                draw_code_with_underlines_scaled(original_code, green_ranges, xfg_nodes_line_number, full_path, flaw_line_index_dict, file_id)






