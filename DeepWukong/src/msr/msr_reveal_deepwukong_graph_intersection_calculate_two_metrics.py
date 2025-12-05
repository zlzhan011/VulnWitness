import pickle

import matplotlib.pyplot as plt
import os
import pandas as pd
import networkx as nx
from tqdm import tqdm
import os
import pandas as pd
from collections import defaultdict
from collections import Counter
import numpy as np
import pandas as pd
import  json


def expand_ranges(ranges):
    """Expand list of {'start_offset': int, 'end_offset': int} to a set of indices."""
    indices = set()
    for r in ranges:
        indices.update(range(r['start_offset'], r['end_offset']))
    return indices


def read_flaw_lines_index():
    c_root = '/scratch/c00590656/vulnerability/LineVul/data/big-vul_dataset'
    c_output = '/scratch/c00590656/vulnerability/DeepWukong/data/msr/source-code/'

    flaw_line_index_dict = {}
    all_c_files_dict = {}
    for file in ['val.csv','test.csv']: # , 'test.csv', 'train.csv'
        file_path = os.path.join(c_root, file)
        file_df = pd.read_csv(file_path)
        print(file_df.columns.values)
        for ii, row in tqdm(file_df.iterrows(), total=len(file_df), desc="Processing rows"):
            func_before = row['func_before']
            func_after = row['func_after']
            CVE_ID = row['CVE ID']
            CWE_ID =  row['CWE ID']
            target = row['target']
            index =  row['index']
            project = row['project'].strip().lower()
            flaw_line_index = row['flaw_line_index']
            all_c_files_dict[str(index)] = func_before

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

    return flaw_line_index_dict, all_c_files_dict



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

    # sorted_tokens = [item for item in sorted_tokens if item['rank_id'] <= 50]

    return sorted_tokens

def get_label_positive_predict_positive_id(linevul_token_scores_dir, file_id_list=[]):

    file_id_list_original = os.listdir(linevul_token_scores_dir)

    file_id_dict = {}
    for item in file_id_list_original:
        file_id_dict[item.split('.')[0]] = item


    file_id_list = [file_id_dict[item] for item in file_id_list]

    linevul_token_scores_dict = rank_for_linevul_tokens(file_id_list, linevul_token_scores_dir)

    return linevul_token_scores_dict


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

    # # 移除 original_order，如果你不需要保留
    # for item in sorted_tokens:
    #     item.pop("original_order")

    # sorted_tokens = [item for item in sorted_tokens if item['rank_id'] <= 100]
    return sorted_tokens




# def calculate_intersection_between_linevul_tokens_socre_and_xfg_nodes(linevul_token_scores, xfg_nodes_line_number_map_offsets):
#     # 输入两个列表
#     xfg_nodes = xfg_nodes_line_number_map_offsets
#     token_scores = linevul_token_scores
#
#     # 初始化结果列表
#     xfg_token_counts = []
#
#     # 遍历每个 xfg node 的 offset 范围
#     for xfg_node in xfg_nodes:
#         xfg_start = xfg_node['start_offset']
#         xfg_end = xfg_node['end_offset']
#         print("\n\n\n*******\nxfg_start:", xfg_start, "xfg_end:", xfg_end)
#         # 统计有多少 token 落在这个范围内
#         count = 0
#         for token in token_scores:
#             tok_start = token.get('start_offset')
#             tok_end = token.get('end_offset')
#
#             # 有些 token 的 start_offset 是 NaN 或 None，要排除
#             if tok_start is None or tok_end is None:
#                 continue
#
#             if xfg_start <= tok_start < xfg_end:
#                 print("tok_start:", tok_start, "tok_end:", tok_end)
#                 count += 1
#
#         xfg_token_counts.append(count)
#
#     # 输出统计结果
#     print("每个 xfg node 中包含的 token 数量：", xfg_token_counts)



#  为什么deepwukong的结果有重复的item?
#  如何设定deepwukong的weight?

def calculate_intersection_between_linevul_tokens_socre_and_xfg_nodes(deepwukong_token_scores, reveal_tokens_score):
    import math
    import pandas as pd

    # 假设已存在两个列表：linevul_token_scores 和 xfg_nodes_line_number_map_offsets
    # Step 1: 清理并排序 token scores，去掉 NaN 的 token

    # tokens = [t for t in linevul_token_scores if isinstance(t.get("rank_id"), int)]
    # tokens = deepwukong_token_scores
    reveal_tokens_score = sorted(reveal_tokens_score, key=lambda x: x["weight"])  # 按照 rank_id 升序排列


    already_item_offset_list = []
    deepwukong_tokens_filtered = []
    for item in  deepwukong_token_scores:
        end_offset = item["end_offset"]
        start_offset = item["start_offset"]
        offset_combine = str(start_offset)+"###"+str(end_offset)
        if offset_combine not in already_item_offset_list:
            already_item_offset_list.append(offset_combine)
            deepwukong_tokens_filtered.append(item)


    deepwukong_tokens_filtered = sorted(deepwukong_tokens_filtered, key=lambda x: x["start_offset"])  # 按照 rank_id 升序排列

    deepwukomg_token_scores_max_end_offset = max([item["end_offset"] for item in deepwukong_tokens_filtered])
    deepwukomg_token_scores_min_start_offset = min([item["start_offset"] for item in deepwukong_tokens_filtered])

    reveal_nodes_line_number_map_offsets_max_end_offset = max([item["end_offset"] for item in reveal_tokens_score])
    reveal_nodes_line_number_map_offsets_min_start_offset = min([item["start_offset"] for item in reveal_tokens_score])


    max_offset = max(deepwukomg_token_scores_max_end_offset, reveal_nodes_line_number_map_offsets_max_end_offset)
    max_offset = max_offset + 1
    deepwukong_token_mask_index = max_offset * [0]



    for item in deepwukong_token_scores:
        start_offset = item["start_offset"]
        end_offset = item["end_offset"]
        for index in range(start_offset, end_offset+1):
            deepwukong_token_mask_index[index] = 1




    reveal_total_tokens = len(reveal_tokens_score)
    bucket_size = math.floor(reveal_total_tokens / 20)  # 10% 分桶，最后一桶可能略小

    results = []



    # Step 2: 遍历每个 10% 区间
    for i in range(20):
        start = i * bucket_size

        if i != 19:
            end = min((i + 1) * bucket_size, reveal_total_tokens)
        else:
            end = reveal_total_tokens

        reveal_tokens_score_bucket = reveal_tokens_score[start:end]

        reveal_nodes_mask_index = max_offset * [0]

        for item in reveal_tokens_score_bucket:
            start_offset = item["start_offset"]
            end_offset = item["end_offset"]
            for index in range(start_offset, end_offset+1):
                reveal_nodes_mask_index[index] = 1


        same_letter_index_cnt = 0
        different_letter_index_cnt = 0
        for index in range(max_offset):
            linevul_token_mask_value = deepwukong_token_mask_index[index]
            xfg_nodes_mask_value = reveal_nodes_mask_index[index]
            if linevul_token_mask_value == xfg_nodes_mask_value and xfg_nodes_mask_value == 1:
                same_letter_index_cnt += 1
            else:
                if xfg_nodes_mask_value == 1 and linevul_token_mask_value == 0:
                    different_letter_index_cnt += 1

        same_letter_rate = same_letter_index_cnt / sum(deepwukong_token_mask_index)
        different_letter_rate = different_letter_index_cnt / sum(deepwukong_token_mask_index)

        results.append({
            'bucket_range': f'{i * 5}-{(i + 1) * 5}%',
            'matched_count': same_letter_index_cnt,
            'unmatched_count': different_letter_index_cnt,
            'matched_ratio': round(same_letter_rate, 4),
            'unmatched_ratio': round(different_letter_rate, 4)
        })
    return results



from pcpp import Preprocessor


def extract_pcpp_tokens(source_code):
    pp = Preprocessor()
    pp.parse(source_code)

    tokens = list(pp.tokenize(source_code))

    parts = [(token.type, token.value) for token in tokens]
    pcpp_tokens = [ token.value for token in tokens]
    return parts, pcpp_tokens



def compute_pcpp_offsets(original_text, pcpp_tokens):
    """
    根据原始代码字符串和 pcpp_tokens 列表，
    计算每个 token 在原始文本中的起始和结束位置。
    返回一个列表，列表中每个元素为 (start, end)。
    """
    offsets = []
    pos = 0
    for token in pcpp_tokens:
        # 注意：token 中可能包含空格或换行，直接查找即可
        idx = original_text.find(token, pos)
        if idx == -1:
            # 如果找不到，记录为 (None, None) 并发出警告
            print(f"Warning 1: 未在原文中找到 token: {repr(token)}")
            offsets.append({"start_offset": None, "end_offset": None + len(token), "token": token})
        else:
            offsets.append({"start_offset":idx, "end_offset":idx + len(token), "token":token})
            pos = idx + len(token)  # 更新搜索起点，保证顺序查找
    return offsets



def count_pcpp_equal_to_linevul_tokenizer(pcpp_tokens_offsets, linevul_tokens_offsets):
    # 假设 green_ranges 和 pcpp_tokens_offsets 已经定义好了

    matched = 0
    unmatched = 0

    # 构建一个集合，加速匹配
    pcpp_offsets_set = set((item['start_offset'], item['end_offset']) for item in pcpp_tokens_offsets)

    for item in linevul_tokens_offsets:
        key = (item['start_offset'], item['end_offset'])
        if key in pcpp_offsets_set:
            matched += 1
        else:
            unmatched += 1

    print("匹配的数量:", matched)
    print("不匹配的数量:", unmatched)

    return matched, unmatched




def analysis_pcpp_bpe_res(pcpp_bpe_matched_res):
    from collections import defaultdict

    # 假设你的列表是 pcpp_bpe_matched_res，包含多个字典，每个字典有 matched_rate 字段
    bucket_stats = defaultdict(int)

    # 遍历所有记录，根据 matched_rate 分桶
    for item in pcpp_bpe_matched_res:
        rate = item["matched_rate"]
        # 将 rate 映射到 10% ~ 100% 的区间段（不包含 0%-10%）
        for i in range(10, 100, 10):
            lower = i / 100
            upper = (i + 10) / 100
            if lower <= rate < upper:
                bucket_stats[f"{i}-{i + 10}%"] += 1
                break
        # 特别处理 100% 的情况
        if rate == 1.0:
            bucket_stats["90-100%"] += 1

    # 总数
    total = len(pcpp_bpe_matched_res)

    # 输出统计结果
    for bucket in sorted(bucket_stats):
        count = bucket_stats[bucket]
        percentage = count / total * 100
        print(f"{bucket}: count = {count}, rate = {percentage:.2f}%")




def is_top_n_bpc_contain_graph_nodes(input_dir, output_dir):

    # 读取所有 Excel 文件
    all_thresholds = []
    for filename in os.listdir(input_dir):
        if not filename.endswith(".xlsx"):
            continue

        filepath = os.path.join(input_dir, filename)
        df = pd.read_excel(filepath)



        bpc_tokens_includes_graph_nodes = "95-100%"
        for ii, row in df.iloc[::-1].iterrows():
            bucket_range = row['bucket_range']
            matched_count = row['matched_count']
            if matched_count == 0:
                bpc_tokens_includes_graph_nodes = bucket_range
            else:
                break
        print("bpc_tokens_includes_graph_nodes:", bpc_tokens_includes_graph_nodes, filename)
        all_thresholds.append(bpc_tokens_includes_graph_nodes)
    count = Counter(all_thresholds)
    print(count)
    # 转为 DataFrame
    df = pd.DataFrame.from_dict(count, orient='index', columns=['count'])
    # 重置索引并命名列
    df = df.reset_index().rename(columns={'index': 'top_n_percent_range'})
    # 可选：按 bucket_range 排序
    df = df.sort_values(by='top_n_percent_range')

    df.to_excel(os.path.join(output_dir, "top_n_bpc_contain_graph_nodes.xlsx"), index=False)
    print(df)





def analysis_range(values, output_dir, file_name='bucket_range_15_above_percent.xlsx'):
    # 定义分区
    bins = np.linspace(0.0, 1.0, 11)  # 0.0 到 1.0 分成 10 个区间
    labels = [f"{round(bins[i], 1)}–{round(bins[i + 1], 1)}" for i in range(10)]

    # 使用 pandas 分箱统计
    bucket = pd.cut(values, bins=bins, labels=labels, include_lowest=True, right=False)
    counts = bucket.value_counts().sort_index()

    # 计算比例
    ratios = (counts / len(values)).round(4)

    # 构造 DataFrame
    df = pd.DataFrame({'range': counts.index, 'count': counts.values, 'ratio': ratios.values})

    print(df)

    df.to_excel(os.path.join(output_dir, file_name), index=False)



def analysis_bpe_tokens_graph_nodes(input_dir, output_dir):
    bucket_range_15_above_percent = []
    bucket_range_25_above_percent = []
    bucket_range_35_above_percent = []
    bucket_range_45_above_percent = []
    bucket_range_95_above_percent = []
    for file in os.listdir(input_dir):
        if 'top' in file:
            continue
        print("file：", file)
        file_path = os.path.join(input_dir, file)
        df = pd.read_excel(file_path)

        matched_count_sum = sum(df['matched_count'])
        matched_count_above_sum = 0
        for ii, row in df.iterrows():
            print("row:", row)
            bucket_range = row['bucket_range']
            matched_count = row['matched_count']
            matched_count_above_sum += matched_count
            if matched_count_sum>0:
                bucket_range_above_percent = matched_count_above_sum/ matched_count_sum
            else:
                bucket_range_above_percent = 0

            if bucket_range == '15-20%':
                bucket_range_15_above_percent.append(bucket_range_above_percent)
            if bucket_range == '25-30%':
                bucket_range_25_above_percent.append(bucket_range_above_percent)
            if bucket_range == '35-40%':
                bucket_range_35_above_percent.append(bucket_range_above_percent)
            if bucket_range == '45-50%':
                bucket_range_45_above_percent.append(bucket_range_above_percent)
            if bucket_range == '90-95%':
                bucket_range_95_above_percent.append(bucket_range_above_percent)

    print("****\nbucket_range_15_above_percent:")
    analysis_range(bucket_range_15_above_percent, output_dir, file_name='bucket_range_15_above_percent.xlsx')
    print("****\nbucket_range_25_above_percent:")
    analysis_range(bucket_range_25_above_percent, output_dir, file_name='bucket_range_25_above_percent.xlsx')
    print("****\nbucket_range_35_above_percent:")
    analysis_range(bucket_range_35_above_percent, output_dir, file_name='bucket_range_35_above_percent.xlsx')
    print("****\nbucket_range_45_above_percent:")
    analysis_range(bucket_range_45_above_percent, output_dir, file_name='bucket_range_45_above_percent.xlsx')

    print("****\nbucket_range_95_above_percent:")
    analysis_range(bucket_range_95_above_percent, output_dir, file_name='bucket_range_95_above_percent.xlsx')







def read_reveal_result():
    reveal_result_dir = '/scratch/c00590656/vulnerability/data-package/interpretability/data_collection/Devign/scripts/explanation_nodes_importance'

    reveal_nodes_weight_importance_info = {}
    for filename in tqdm(os.listdir(reveal_result_dir), desc="Processing files"):
        if 'id_' in filename:
            continue

        file_id = filename.split('_')[0]
        file_path = os.path.join(reveal_result_dir, filename)

        # 如果文件大小为 0，直接跳过
        if os.path.getsize(file_path) == 0:
            # print(f"跳过空文件: {file_path}")
            continue

        # 尝试读取 JSON，如果解析失败就跳过
        try:
            with open(file_path, 'r') as f:
                json_data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # print(f"跳过无法解析的文件: {file_path}")
            continue

        # 如果到这里说明 json_data 正常
        nodes_weight_importance_info = json_data.get('nodes_weight_importance_info', None)
        if nodes_weight_importance_info is None:
            # 如果 JSON 里没有期望的字段，也跳过
            continue

        # 把原来的格式转换成我们需要的形式
        nodes_weight_importance_info_new = []
        for item in nodes_weight_importance_info:
            # item['code'], item['start_idx'], item['end_idx'], item['score']
            nodes_weight_importance_info_new.append({
                "token":     item['code'],
                "start_offset": int(item['start_idx']),
                "end_offset":   int(item['end_idx']),
                "weight":    item['score'],
                "node_idx":  item['node_idx']
            })

        reveal_nodes_weight_importance_info[file_id] = nodes_weight_importance_info_new

    return reveal_nodes_weight_importance_info





def read_deepwukong_graph_nodes(dirpath, filename):
    full_path = os.path.join(dirpath, filename)
    one_file_id_files.append(full_path)
    label_slicing, xfg = read_xfg_label(full_path)
    # print("label_slicing:", label_slicing)
    if label_slicing == 1 or (label_slicing == 0 and 'vulnerable' in full_path):
        pass
    else:
        return None

    print("full_path:", full_path)
    # print("label_slicing:", label_slicing)
    import ast
    import json

    xfg_nodes_line_number = list(xfg.nodes)
    line_number_offsets = get_line_offsets(original_code)

    if len(xfg_nodes_line_number) == 0:
        return None

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
    black_ranges = xfg_nodes_line_number_map_offsets

    return black_ranges





def read_deepwukong_slicing(c_root, file_id, all_c_files_dict):

    # deepwukong_slicing_res = {}
    # for file_id in os.listdir(c_root):

    deepwukong_slicing_res ={}
    if file_id not in all_c_files_dict:
        return []
    original_code = all_c_files_dict[file_id]
    file_id_dir = os.path.join(c_root, file_id)

    xfg_nodes_line_number_map_offsets = []
    for dirpath, dirnames, filenames in os.walk(file_id_dir):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            label_slicing, xfg = read_xfg_label(full_path)
            # print("label_slicing:", label_slicing)
            if label_slicing == 1 or (label_slicing == 0 and 'vulnerable' in full_path):
                pass
            else:
                continue

            print("full_path:", full_path)


            xfg_nodes_line_number = list(xfg.nodes)
            line_number_offsets = get_line_offsets(original_code)

            if len(xfg_nodes_line_number) == 0:
                continue

            for one_line_number in xfg_nodes_line_number:
                line_start_index = line_number_offsets[one_line_number]['start_offset']
                line_end_index = line_number_offsets[one_line_number]['end_offset']
                wight = 1
                one_node = {"start_offset": int(line_start_index),
                            "end_offset": int(line_end_index),
                            "token": line_number_offsets[one_line_number]['code'],
                            "wight": wight, }
                xfg_nodes_line_number_map_offsets.append(one_node)



            deepwukong_slicing_res[file_id] = xfg_nodes_line_number_map_offsets

    return xfg_nodes_line_number_map_offsets

if __name__ == '__main__':

    root = '/scratch/c00590656/vulnerability/DeepWukong/data/msr/source-code/'



    c_root = '/scratch/c00590656/vulnerability/DeepWukong/data/msr/XFG_LineVul_Map'
    c_root = '/scratch/c00590656/vulnerability/DeepWukong/src/data/msr/XFG_LineVul_Map'
    c_root = '/scratch/c00590656/vulnerability/DeepWukong/src/data/msr/XFG'
    output_dir = '/scratch/c00590656/vulnerability/DeepWukong/src/msr/msr_reveal_graph_linevul_intersection_last_step_analysis_res'

    linevul_token_scores_dir = '/scratch/c00590656/vulnerability/LineVul/linevul/shap/saved_models_MSR/token_weight_map'
    linevul_token_scores_dir = '/scratch/c00590656/vulnerability/LineVul/linevul/shap/saved_models_MSR/token_weight_map_predict_positive_label_positive'
    file_id_list_original = os.listdir(linevul_token_scores_dir)
    file_id_list = [item.split('.')[0] for item in file_id_list_original]

    analysis_bpe_tokens_graph_nodes(os.path.join(output_dir, 'all_files_intersection_res_two_graphs/each_instances_res'), os.path.join(output_dir, 'all_files_intersection_res_two_graphs'))
    exit()


    # is_top_n_bpc_contain_graph_nodes( os.path.join(output_dir, 'all_files_intersection_res_two_graphs/each_instances_res'), os.path.join(output_dir, 'all_files_intersection_res_two_graphs'))
    # exit()


    reveal_nodes_weight_importance_info = read_reveal_result()

    flaw_line_index_dict, all_c_files_dict = read_flaw_lines_index()





    root_dir_list = os.listdir(c_root)
    ii = 0
    top_k_tokens = []
    all_files_intersection_res_two_graphs = []
    pcpp_bpe_matched_res = []


    for file_id in tqdm(root_dir_list, desc="Processing files"): # file_id_list


        if file_id not in file_id_list:
            continue

        deepwukong_slicing_res = read_deepwukong_slicing(c_root, file_id,all_c_files_dict)

        # linevul_token_scores_dict = get_label_positive_predict_positive_id(linevul_token_scores_dir, file_id_list = [str(file_id)])

        print("\n\nfile_id:", file_id)

        original_code = all_c_files_dict[file_id]



        if len(deepwukong_slicing_res) == 0:
            continue


        if str(file_id) in reveal_nodes_weight_importance_info:
            black_ranges = reveal_nodes_weight_importance_info[str(file_id)]
            ii = ii + 1
            print("matched ii:", ii)
        else:
            continue

        one_files_intersection_res = calculate_intersection_between_linevul_tokens_socre_and_xfg_nodes(deepwukong_slicing_res,
                                                                          reveal_nodes_weight_importance_info[str(file_id)])



        all_files_intersection_res_two_graphs = all_files_intersection_res_two_graphs + one_files_intersection_res
        one_files_intersection_res_df = pd.DataFrame(one_files_intersection_res)

        one_files_intersection_res_df.to_excel(os.path.join(output_dir, 'all_files_intersection_res_two_graphs/each_instances_res', file_id+".xlsx"), index=False)

        print("one_files_intersection_res_df:", one_files_intersection_res_df)
    print("all_files_intersection_res_two_graphs_len:", len(all_files_intersection_res_two_graphs))

    analysis_pcpp_bpe_res(pcpp_bpe_matched_res)
    pcpp_bpe_matched_res_df = pd.DataFrame(pcpp_bpe_matched_res)
    pcpp_bpe_matched_res_df.to_excel(os.path.join(output_dir ,"all_files_intersection_res_two_graphs", 'pcpp_bpe_matched_res_df.xlsx'))




    all_files_intersection_res_two_graphs = pd.DataFrame(all_files_intersection_res_two_graphs)
    all_files_intersection_res_two_graphs.to_excel(os.path.join(output_dir ,"all_files_intersection_res_two_graphs", "all_files_intersection_res_two_graphs.xlsx"), index=False)

    # Step 2: 按 bucket_range 分组，汇总 matched_count 和 unmatched_count
    agg_df = all_files_intersection_res_two_graphs.groupby("bucket_range")[["matched_count", "unmatched_count"]].sum().reset_index()

    # Step 3: 计算新的 matched_ratio 和 unmatched_ratio
    agg_df["matched_ratio"] = agg_df["matched_count"] / (agg_df["matched_count"] + agg_df["unmatched_count"])
    agg_df["unmatched_ratio"] = 1 - agg_df["matched_ratio"]
    # print("agg_df:", agg_df)
    agg_df.to_excel(os.path.join(output_dir ,"all_files_intersection_res_two_graphs", "all_files_intersection_res_two_graphs_agg_analysis_res.xlsx"), index=False)







