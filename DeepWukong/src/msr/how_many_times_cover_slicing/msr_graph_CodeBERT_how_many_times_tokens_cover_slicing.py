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
import json
# ---- 在导入 transformers 之前加上这段 ----
import importlib.metadata as _im

_real_version = _im.version

def _safe_version(name: str) -> str:
    """
    包含非法 metadata（缺少 Version 字段）时，
    避免 KeyError，直接返回一个占位版本号。
    """
    try:
        return _real_version(name)
    except KeyError:
        # 某些损坏的包 metadata 没有 Version 字段，返回一个 dummy 版本即可
        return "99.0.0"

_im.version = _safe_version
# ---- 补丁结束 ----

# 然后再正常导入 transformers
from transformers import RobertaTokenizer

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

    # sorted_tokens = [item for item in sorted_tokens if item['rank_id'] <= 50]

    return sorted_tokens

def clean_utf8(token: str) -> str:
    """
    按你的要求：
    - 对 token 中每个字符逐个判断
    - 可 UTF-8 编码则保留
    - 不可编码则丢弃（替换为空字符串）
    """
    cleaned = []
    for ch in token:
        try:
            ch.encode("utf-8")
            cleaned.append(ch)
        except UnicodeEncodeError:
            # 跳过该字符（你的需求）
            pass
    return ''.join(cleaned)



def _is_valid_utf8_char(c):
    try:
        c.encode("utf-8")
        return True
    except:
        return False


def align_tokens_with_offsets(text, tokens, tokenizer):
    """
    对齐 CodeBERT/BPE tokens 到原始 text 的字符偏移（使用 find）。
    你要求的 UTF-8 过滤也保留。
    """

    result = []
    cursor = 0
    text_len = len(text)

    for token in tokens:

        # 过滤掉 decode 后为空的 token （例如 'Ġ', 控制符等）
        decoded_token = tokenizer.convert_tokens_to_string([token])


        # 清理 UTF-8 不支持字符
        decoded_token = clean_utf8(decoded_token)

        if decoded_token.strip() == "":
            result.append({
                "token": decoded_token,
                "start_offset": None,
                "end_offset": None
            })
            continue

        # 使用 find 从 cursor 开始查找 token
        pos = text.find(decoded_token, cursor)

        if pos == -1:
            # 找不到，记录 None
            result.append({
                "token": decoded_token,
                "start_offset": None,
                "end_offset": None
            })
            continue

        # 找到 offset
        start_offset = pos
        end_offset = pos + len(decoded_token)

        result.append({
            "token": decoded_token,
            "start_offset": start_offset,
            "end_offset": end_offset
        })

        # cursor 移到当前 token 后面继续找
        cursor = end_offset

    return result

def rank_for_linevul_tokens_from_shap(file_id_list, shap_token_scores_dir):
    """
    从 shap_token_weights 目录下的 pkl 里读取：
        - tokens
        - token_weights
    组装成和原来 Excel 读出来一样的结构：
        [
            {
                "token": token,
                "weight": weight,
                "start_offset": None,
                "end_offset": None,
                "rank_id": ...
            },
            ...
        ]
    并且只保留 label=1 且预测为 positive 的样本。
    """
    linevul_token_scores_dict = {}

    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    for file_name in file_id_list:
        if not file_name.endswith(".pkl"):
            continue

        file_path = os.path.join(shap_token_scores_dir, file_name)
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        # # 使用封装后的判断函数
        # if not is_label_positive_and_pred_positive(data):
        #     continue


        tokens = data["tokens"]
        weights = data["token_weights"]
        offset_info = align_tokens_with_offsets(data['text'], tokens, tokenizer)
        assert len(tokens) == len(weights), f"tokens/weights 长度不一致: {file_name}"
        assert len(offset_info) == len(tokens), f"tokens/offset_info 长度不一致: {file_name}"
        one_file_token_information_list = []
        for tok, w, offset in zip(tokens, weights, offset_info):
            one_file_token_information_list.append(
                {
                    "token": offset['token'],
                    "weight": w,
                    # 目前 pkl 里没有 offset 信息，用 None 占位，保持字段名一致
                    "start_offset": offset['start_offset'],
                    "end_offset": offset['end_offset'],
                }
            )

        # 排序加 rank_id
        one_file_token_information_list = sort_tokens_and_get_rank_id(one_file_token_information_list)

        # key 用不带后缀的 file_id（和你原来 file_id_list 一致）
        file_id_no_ext = file_name.split(".")[0]
        linevul_token_scores_dict[file_id_no_ext] = one_file_token_information_list

    return linevul_token_scores_dict

def get_label_positive_predict_positive_id(shap_token_scores_dir):
    """
    从新的 SHAP pkl 目录里提取：
        - file_id_list: 不带后缀的 id 列表（只包含 label=1 且预测为 positive 的）
        - linevul_token_scores_dict: {file_id: [token_info_dict, ...]}
    返回格式与原 get_label_positive_predict_positive_id 一致。
    """

    file_id_list_original = [
        f for f in os.listdir(shap_token_scores_dir)
        if f.endswith(".pkl")
    ]

    # rank + 过滤 label/predict
    linevul_token_scores_dict = rank_for_linevul_tokens_from_shap(
        file_id_list_original,
        shap_token_scores_dir
    )

    # 只保留真正通过过滤、且在 dict 里的那些 id
    file_id_list = list(linevul_token_scores_dict.keys())

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

    # # 移除 original_order，如果你不需要保留
    # for item in sorted_tokens:
    #     item.pop("original_order")

    # sorted_tokens = [item for item in sorted_tokens if item['rank_id'] <= 100]
    return sorted_tokens










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
    df = df.reset_index().rename(columns={'index': 'bucket_range'})
    # 可选：按 bucket_range 排序
    df = df.sort_values(by='bucket_range')

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

    print("****\nbucket_range_15_above_percent:")
    analysis_range(bucket_range_15_above_percent, output_dir, file_name='bucket_range_15_above_percent.xlsx')
    print("****\nbucket_range_25_above_percent:")
    analysis_range(bucket_range_25_above_percent, output_dir, file_name='bucket_range_25_above_percent.xlsx')
    print("****\nbucket_range_35_above_percent:")
    analysis_range(bucket_range_35_above_percent, output_dir, file_name='bucket_range_35_above_percent.xlsx')
    print("****\nbucket_range_45_above_percent:")
    analysis_range(bucket_range_45_above_percent, output_dir, file_name='bucket_range_45_above_percent.xlsx')




def calculate_intersection_between_linevul_tokens_socre_and_xfg_nodes(
        linevul_token_scores,
        xfg_nodes_line_number_map_offsets
):
    """
    对于一个函数中的一个 slicing（由 xfg_nodes_line_number_map_offsets 描述）：

    1. 找出所有落在 slicing 范围内的 linevul tokens（形成 slicing_token_indices 集合）
    2. 已知 linevul_token_scores 是按权重从高到低排序的列表：
       - 记 slicing_tokens_number = len(slicing_token_indices)
       - 取前 1×slicing_tokens_number 个 tokens，计算：
           * jaccard_1x
           * covered_tokens_1x / missed_tokens_1x / missed_ratio_1x
       - 取前 2×slicing_tokens_number 个 tokens（同上）
       - 取前 3×slicing_tokens_number 个 tokens（同上）
       - 取前 4×slicing_tokens_number 个 tokens（同上）
       - 取“全部” tokens（同上）
       - 从 top-1 开始依次加入 token，直到覆盖所有 slicing_token_indices，
         得到“until_cover” 情景，并计算对应的：
           * X_times_LineVul_tokens_cover_slicing
           * jaccard_until_cover
           * covered_tokens_until_cover / missed_tokens_until_cover / missed_ratio_until_cover
    3. 返回一个 dict，包含上述所有信息。
    """

    total_tokens = len(linevul_token_scores)

    # ---------- 1) 找出 slicing 中包含的 tokens（用索引来表示每个 token 的“id”） ----------
    slicing_token_indices = set()
    for idx, tok in enumerate(linevul_token_scores):
        t_start = tok.get('start_offset')
        t_end = tok.get('end_offset')

        # 过滤掉 NaN 或 None 的 offset
        if t_start is None or t_end is None:
            continue
        if isinstance(t_start, float) and pd.isna(t_start):
            continue
        if isinstance(t_end, float) and pd.isna(t_end):
            continue

        for node in xfg_nodes_line_number_map_offsets:
            xfg_start = node['start_offset']
            xfg_end = node['end_offset']
            # 只要整个 token 落在某个 slicing node 的范围内，就认为该 token 属于 slicing
            if xfg_start <= t_start and t_end <= xfg_end:
                slicing_token_indices.add(idx)
                break

    slicing_tokens_number = len(slicing_token_indices)

    # 如果这个 slicing 里根本没有覆盖到任何 token，就返回一个特殊结果
    if slicing_tokens_number == 0 or total_tokens == 0:
        return {
            "slicing_tokens_number": 0,
            "selected_linevul_tokens_number": 0,
            "X_times_LineVul_tokens_cover_slicing": None,
            "jaccard": 0.0,  # 原有字段，保留
            "intersection_size": 0,
            "union_size": 0,
            "is_coverage_reached": False,

            # 6 种情景：选中 token 数
            "selected_tokens_1x": 0,
            "selected_tokens_2x": 0,
            "selected_tokens_3x": 0,
            "selected_tokens_4x": 0,
            "selected_tokens_all": 0,
            "selected_tokens_until_cover": 0,

            # 6 种情景：X 倍覆盖
            "X_times_LineVul_tokens_cover_slicing_1x": None,
            "X_times_LineVul_tokens_cover_slicing_2x": None,
            "X_times_LineVul_tokens_cover_slicing_3x": None,
            "X_times_LineVul_tokens_cover_slicing_4x": None,
            "X_times_LineVul_tokens_cover_slicing_all": None,
            "X_times_LineVul_tokens_cover_slicing_until_cover": None,

            # 6 种情景：Jaccard
            "jaccard_1x": 0.0,
            "jaccard_2x": 0.0,
            "jaccard_3x": 0.0,
            "jaccard_4x": 0.0,
            "jaccard_all": 0.0,
            "jaccard_until_cover": 0.0,

            # 6 种情景：覆盖 / 未覆盖数量 & 占比
            "covered_tokens_1x": 0,
            "missed_tokens_1x": 0,
            "missed_ratio_1x": None,

            "covered_tokens_2x": 0,
            "missed_tokens_2x": 0,
            "missed_ratio_2x": None,

            "covered_tokens_3x": 0,
            "missed_tokens_3x": 0,
            "missed_ratio_3x": None,

            "covered_tokens_4x": 0,
            "missed_tokens_4x": 0,
            "missed_ratio_4x": None,

            "covered_tokens_all": 0,
            "missed_tokens_all": 0,
            "missed_ratio_all": None,

            "covered_tokens_until_cover": 0,
            "missed_tokens_until_cover": 0,
            "missed_ratio_until_cover": None,
        }

    # ---------- 工具函数：给定选中的前 K 个 token，计算 Jaccard & 未覆盖比例 ----------
    def calc_metrics_for_top_k(k):
        if k <= 0:
            return 0.0, 0, None, 0, slicing_tokens_number, None
        k = min(k, total_tokens)
        selected_indices = set(range(k))
        inter = len(selected_indices & slicing_token_indices)
        union = len(selected_indices | slicing_token_indices)
        j = inter / union if union > 0 else 0.0
        X_times_LineVul_tokens_cover_slicing_k = k / slicing_tokens_number
        missed = slicing_tokens_number - inter
        missed_ratio = missed / slicing_tokens_number if slicing_tokens_number > 0 else None
        return j, k, X_times_LineVul_tokens_cover_slicing_k, inter, missed, missed_ratio

    # ---------- 2) 1×/2×/3×/4×/all 情景 ----------
    n = slicing_tokens_number

    j_1x, sel_1x, cov_1x, inter_1x, miss_1x, miss_ratio_1x = calc_metrics_for_top_k(1 * n)
    j_2x, sel_2x, cov_2x, inter_2x, miss_2x, miss_ratio_2x = calc_metrics_for_top_k(2 * n)
    j_3x, sel_3x, cov_3x, inter_3x, miss_3x, miss_ratio_3x = calc_metrics_for_top_k(3 * n)
    j_4x, sel_4x, cov_4x, inter_4x, miss_4x, miss_ratio_4x = calc_metrics_for_top_k(4 * n)
    j_all, sel_all, cov_all, inter_all, miss_all, miss_ratio_all = calc_metrics_for_top_k(total_tokens)

    # ---------- 3) until_cover 情景（原有逻辑） ----------
    selected_indices_until = set()
    coverage_reached = False
    selected_cnt_until = 0

    for idx in range(total_tokens):
        selected_indices_until.add(idx)
        if slicing_token_indices.issubset(selected_indices_until):
            coverage_reached = True
            selected_cnt_until = idx + 1
            break

    if not coverage_reached:
        # 没有完全覆盖，就直接认为选了全部 tokens
        selected_cnt_until = total_tokens

    X_times_LineVul_tokens_cover_slicing_until = selected_cnt_until / slicing_tokens_number

    inter_until = len(selected_indices_until & slicing_token_indices)
    union_until = len(selected_indices_until | slicing_token_indices)
    j_until = inter_until / union_until if union_until > 0 else 0.0

    missed_until = slicing_tokens_number - inter_until
    missed_ratio_until = missed_until / slicing_tokens_number if slicing_tokens_number > 0 else None

    # ---------- 为了兼容旧代码，保留原有字段名 ----------
    # 原来的 X_times_LineVul_tokens_cover_slicing / jaccard 就对应 “until_cover” 情景
    return {
        "slicing_tokens_number": slicing_tokens_number,

        # 旧字段：对应 until_cover 情景
        "selected_linevul_tokens_number": selected_cnt_until,
        "X_times_LineVul_tokens_cover_slicing": X_times_LineVul_tokens_cover_slicing_until,
        "jaccard": j_until,
        "intersection_size": inter_until,
        "union_size": union_until,
        "is_coverage_reached": coverage_reached,

        # 6 种情景：选中 token 数
        "selected_tokens_1x": sel_1x,
        "selected_tokens_2x": sel_2x,
        "selected_tokens_3x": sel_3x,
        "selected_tokens_4x": sel_4x,
        "selected_tokens_all": sel_all,
        "selected_tokens_until_cover": selected_cnt_until,

        # 6 种情景：X 倍覆盖
        "X_times_LineVul_tokens_cover_slicing_1x": cov_1x,
        "X_times_LineVul_tokens_cover_slicing_2x": cov_2x,
        "X_times_LineVul_tokens_cover_slicing_3x": cov_3x,
        "X_times_LineVul_tokens_cover_slicing_4x": cov_4x,
        "X_times_LineVul_tokens_cover_slicing_all": cov_all,
        "X_times_LineVul_tokens_cover_slicing_until_cover": X_times_LineVul_tokens_cover_slicing_until,

        # 6 种情景：Jaccard
        "jaccard_1x": j_1x,
        "jaccard_2x": j_2x,
        "jaccard_3x": j_3x,
        "jaccard_4x": j_4x,
        "jaccard_all": j_all,
        "jaccard_until_cover": j_until,

        # 6 种情景：覆盖 / 未覆盖数量 & 占比
        "covered_tokens_1x": inter_1x,
        "missed_tokens_1x": miss_1x,
        "missed_ratio_1x": miss_ratio_1x,

        "covered_tokens_2x": inter_2x,
        "missed_tokens_2x": miss_2x,
        "missed_ratio_2x": miss_ratio_2x,

        "covered_tokens_3x": inter_3x,
        "missed_tokens_3x": miss_3x,
        "missed_ratio_3x": miss_ratio_3x,

        "covered_tokens_4x": inter_4x,
        "missed_tokens_4x": miss_4x,
        "missed_ratio_4x": miss_ratio_4x,

        "covered_tokens_all": inter_all,
        "missed_tokens_all": miss_all,
        "missed_ratio_all": miss_ratio_all,

        "covered_tokens_until_cover": inter_until,
        "missed_tokens_until_cover": missed_until,
        "missed_ratio_until_cover": missed_ratio_until,
    }



def analyze_jaccard_multi_scenarios_vulnerable_only(all_files_intersection_res, output_dir):
    """
    在 xfg_file 包含 'vulnerable' 的 slicing 上，统计 6 种情景下的 jaccard 分布：
      1x, 2x, 3x, 4x, all, until_cover(原 jaccard)
    每个 jaccard bin (0-0.1, 0.1-0.2, ..., 0.9-1.0) 上画 6 根柱子。

    结果：
      - 保存一个 CSV，记录每个 bin 下各情景的百分比
      - 保存一张分组柱状图 PNG
    """
    save_dir = os.path.join(output_dir, "all_files_intersection_res_CodeBERT_DeepWuKong")
    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame(all_files_intersection_res)

    if "xfg_file" not in df.columns:
        print("[WARN] Column 'xfg_file' not found, skip multi-scenario jaccard analysis.")
        return

    mask = df["xfg_file"].astype(str).str.contains("vulnerable", na=False)
    vdf = df[mask].copy()

    if len(vdf) == 0:
        print("[WARN] No vulnerable slicings for multi-scenario jaccard analysis.")
        return

    print(f"[INFO] Multi-scenario jaccard: using {len(vdf)} vulnerable slicings.")

    scenario_cols = {
        "jaccard_1x": "1× slicing tokens",
        "jaccard_2x": "2× slicing tokens",
        "jaccard_3x": "3× slicing tokens",
        "jaccard_4x": "4× slicing tokens",
        "jaccard_all": "All tokens",
        "jaccard": "Until cover",
    }

    bin_edges = np.linspace(0.0, 1.0, 11)
    bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(10)]

    scenario_percentages = {}
    scenario_counts = {}

    for col, display_name in scenario_cols.items():
        if col not in vdf.columns:
            print(f"[WARN] Column '{col}' not found in vulnerable df, skip this scenario.")
            continue

        vals_raw = vdf[col].dropna()
        vals = vals_raw[(vals_raw >= 0.0) & (vals_raw <= 1.0)]
        filtered_out = len(vals_raw) - len(vals)
        if filtered_out > 0:
            print(f"[INFO] Scenario {col}: {filtered_out} values out of [0,1] are filtered out.")

        if len(vals) == 0:
            print(f"[WARN] Scenario {col}: no valid jaccard values, skip.")
            continue

        counts, _ = np.histogram(vals, bins=bin_edges)
        total = counts.sum()
        if total == 0:
            percentages = np.zeros_like(counts, dtype=float)
        else:
            percentages = counts / total * 100.0

        scenario_counts[display_name] = counts
        scenario_percentages[display_name] = percentages

    if not scenario_percentages:
        print("[WARN] No scenario has valid jaccard values. Abort plotting.")
        return

    # 保存 CSV（宽表）
    data_rows = []
    for i, label in enumerate(bin_labels):
        row = {"bin_label": label}
        for display_name, perc_arr in scenario_percentages.items():
            row[display_name] = perc_arr[i]
        data_rows.append(row)

    out_df = pd.DataFrame(data_rows)
    csv_path = os.path.join(save_dir, "jaccard_multi_scenarios_vulnerable_only.csv")
    out_df.to_csv(csv_path, index=False)
    print(f"[INFO] Multi-scenario jaccard distribution CSV saved to: {csv_path}")

    # 分组柱状图（放大字体）
    plt.figure(figsize=(12, 6))
    x = np.arange(len(bin_labels))
    scenario_names = list(scenario_percentages.keys())
    num_scenarios = len(scenario_names)
    bar_width = 0.8 / num_scenarios

    for i, scen in enumerate(scenario_names):
        offset = (i - num_scenarios / 2) * bar_width + bar_width / 2
        plt.bar(
            x + offset,
            scenario_percentages[scen],
            width=bar_width,
            label=scen
        )

    plt.xticks(x, bin_labels, rotation=45, ha="right", fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlabel("Jaccard bins", fontsize=25)
    plt.ylabel("Percentage of instances (%)", fontsize=25)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    # 图例字体放大
    plt.legend(title="Scenario", fontsize=14, title_fontsize=14)

    png_path = os.path.join(save_dir, "jaccard_multi_scenarios_vulnerable_only.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"[INFO] Multi-scenario jaccard histogram saved to: {png_path}")




import os
import pandas as pd
from tqdm import tqdm


# ========== 路径与目录初始化 ==========

def init_paths_and_dirs():
    """初始化路径配置并创建必要的输出目录。"""
    root = '/scratch/c00590656/vulnerability/DeepWukong/data/msr/source-code/'
    c_root = '/scratch/c00590656/vulnerability/DeepWukong/src/data/msr/XFG'
    output_dir = '/scratch/c00590656/vulnerability/DeepWukong/src/msr/msr_graph_linevul_intersection_last_step_analysis_res'
    # linevul_token_scores_dir = '/scratch/c00590656/vulnerability/LineVul/linevul/shap/saved_models_MSR/token_weight_map_predict_positive_label_positive'
    linevul_token_scores_dir = "/scratch/c00590656/vulnerability/data-package/models/CodeBERT/scripts/saved_models_msr_processed_func/shap_token_weights"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "all_files_intersection_res_CodeBERT_DeepWuKong"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "all_files_intersection_res_CodeBERT_DeepWuKong/each_instances_res"), exist_ok=True)

    return root, c_root, output_dir, linevul_token_scores_dir


def get_linevul_file_ids(linevul_token_scores_dir):
    """从 token 权重目录中解析出所有可用的 file_id（不带扩展名）。"""
    file_id_list_original = os.listdir(linevul_token_scores_dir)
    file_id_list = [item.split('.')[0] for item in file_id_list_original]
    return set(file_id_list)


# ========== 单文件处理相关 ==========

def load_original_code(all_c_files_dict, file_id):
    """读取某个 file_id 对应的 C 源码。"""
    c_file_path = all_c_files_dict[file_id]
    with open(c_file_path, "r") as f:
        original_code = f.read()
    return original_code


def load_linevul_scores_for_file(linevul_token_scores_dir, file_id):
    """
    读取并排序某个 file_id 对应的 LineVul token 权重列表。
    返回：linevul_token_scores_list
    """
    file_id_list, linevul_token_scores_dict = get_label_positive_predict_positive_id(
        linevul_token_scores_dir
    )
    return linevul_token_scores_dict[file_id]


def compute_pcpp_bpe_stats(original_code, linevul_token_scores_for_file):
    """
    计算 PCPP tokenizer 与 LineVul tokenizer 之间的匹配情况。
    返回：pcpp_bpe_matched, pcpp_bpe_unmatched, matched_rate
    """
    parts, pcpp_tokens = extract_pcpp_tokens(original_code)
    pcpp_tokens_offsets = compute_pcpp_offsets(original_code, pcpp_tokens)

    matched, unmatched = count_pcpp_equal_to_linevul_tokenizer(
        pcpp_tokens_offsets,
        linevul_token_scores_for_file
    )

    total = matched + unmatched
    matched_rate = matched / total if total > 0 else 0.0

    return matched, unmatched, matched_rate


def iterate_xfg_slicings_for_file(
    file_id,
    c_root,
    original_code,
    linevul_token_scores,
    output_dir
):
    """
    遍历某个函数 file_id 对应的所有 XFG slicing 文件，
    对每个 slicing 计算 coverage / Jaccard 等指标。
    返回：slicing_metrics_list（列表，每个元素是一个 dict）
    """
    file_id_dir = os.path.join(c_root, file_id)
    slicing_metrics_list = []

    # 预先计算每一行的 offset
    line_number_offsets = get_line_offsets(original_code)

    for dirpath, dirnames, filenames in os.walk(file_id_dir):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)

            label_slicing, xfg = read_xfg_label(full_path)
            # 只保留 vulnerable slicing
            # if not (label_slicing == 1 or (label_slicing == 0 and 'vulnerable' in full_path)):
            #     continue

            if 'vulnerable' not in full_path:
                continue

            print("full_path:", full_path)

            xfg_nodes_line_number = list(xfg.nodes)
            if len(xfg_nodes_line_number) == 0:
                continue

            # 构造 slicing 行 -> offset 范围
            xfg_nodes_line_number_map_offsets = []
            for one_line_number in xfg_nodes_line_number:
                line_start_index = line_number_offsets[one_line_number]['start_offset']
                line_end_index = line_number_offsets[one_line_number]['end_offset']
                one_node = {
                    "start_offset": int(line_start_index),
                    "end_offset": int(line_end_index),
                    "token": line_number_offsets[one_line_number]['code'],
                    "wight": 1,
                }
                xfg_nodes_line_number_map_offsets.append(one_node)

            # === 关键：计算当前 slicing 的 coverage 和 Jaccard ===
            metrics = calculate_intersection_between_linevul_tokens_socre_and_xfg_nodes(
                linevul_token_scores,
                xfg_nodes_line_number_map_offsets
            )

            # 增加上下文信息，方便后续分析
            metrics["file_id"] = file_id
            metrics["xfg_file"] = full_path
            metrics["label_slicing"] = label_slicing

            slicing_metrics_list.append(metrics)

            # 单个 slicing 也存一份 Excel（如果需要保留细粒度）
            one_files_intersection_res_df = pd.DataFrame([metrics])
            out_name = f"{file_id}_{os.path.basename(full_path)}.xlsx"
            one_files_intersection_res_df.to_excel(
                os.path.join(
                    output_dir,
                    'all_files_intersection_res/each_instances_res',
                    out_name
                ),
                index=False
            )

    return slicing_metrics_list


def process_one_file(
    file_id,
    all_c_files_dict,
    c_root,
    linevul_token_scores,
    output_dir
):
    """
    处理单个 file_id：
      - 加载源码
      - 加载并排序该文件的 LineVul token 权重
      - 计算 PCPP vs BPE 匹配统计
      - 遍历该函数的所有 XFG slicing，计算 coverage / Jaccard
    返回：
      - pcpp_record: dict（该函数级别的 PCPP/BPE 统计）
      - slicing_metrics_list: list[dict]（所有 slicing 的覆盖/Jaccard 结果）
    """

    # 1. 加载源码 & token 权重
    original_code = load_original_code(all_c_files_dict, file_id)


    # # 2. PCPP vs BPE 统计
    # pcpp_bpe_matched, pcpp_bpe_unmatched, matched_rate = compute_pcpp_bpe_stats(
    #     original_code,
    #     linevul_token_scores_dict
    # )
    # pcpp_record = {
    #     "file_id": file_id,
    #     "pcpp_bpe_matched": pcpp_bpe_matched,
    #     "pcpp_bpe_unmatched": pcpp_bpe_unmatched,
    #     "matched_rate": matched_rate,
    # }

    # 3. 遍历所有 slicing，计算 coverage / Jaccard
    slicing_metrics_list = iterate_xfg_slicings_for_file(
        file_id=file_id,
        c_root=c_root,
        original_code=original_code,
        linevul_token_scores=linevul_token_scores,
        output_dir=output_dir
    )

    return "", slicing_metrics_list


# ========== 结果保存 ==========

def save_all_results(
    output_dir,
    pcpp_bpe_matched_res,
    all_files_intersection_res
):
    """统一保存 PCPP/BPE & slicing 覆盖/Jaccard 的结果到 Excel。"""
    # # 保存 PCPP vs BPE 匹配结果
    # analysis_pcpp_bpe_res(pcpp_bpe_matched_res)
    # pcpp_bpe_matched_res_df = pd.DataFrame(pcpp_bpe_matched_res)
    # pcpp_bpe_matched_res_df.to_excel(
    #     os.path.join(output_dir, "all_files_intersection_res_CodeBERT_DeepWuKong", 'pcpp_bpe_matched_res_df.xlsx'),
    #     index=False
    # )

    # 保存所有 slicing 的 coverage / Jaccard 结果
    all_files_intersection_res_df = pd.DataFrame(all_files_intersection_res)
    all_files_intersection_res_df.to_excel(
        os.path.join(output_dir, "all_files_intersection_res_CodeBERT_DeepWuKong", "coverage_jaccard_res.xlsx"),
        index=False
    )


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def analyze_X_times_LineVul_tokens_cover_slicing_distribution(all_files_intersection_res, output_dir, bins=None):
    """
    统计 X_times_LineVul_tokens_cover_slicing 的分布：
      - 第 1 个柱子：0 < X_times_LineVul_tokens_cover_slicing <= 1
      - 第 2 个柱子：1 < X_times_LineVul_tokens_cover_slicing <= 1.5
      - 第 3 个柱子：1.5 < X_times_LineVul_tokens_cover_slicing <= 2
      - 第 4 个柱子：2 < X_times_LineVul_tokens_cover_slicing <= 3
      - ...
      - 第 11 个柱子：9 < X_times_LineVul_tokens_cover_slicing <= 10
      - 第 12 个柱子：X_times_LineVul_tokens_cover_slicing > 10
    """

    save_dir = os.path.join(output_dir, "all_files_intersection_res_CodeBERT_DeepWuKong")
    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame(all_files_intersection_res)

    if "X_times_LineVul_tokens_cover_slicing" not in df.columns:
        print("[WARN] X_times_LineVul_tokens_cover_slicing 列不存在，无法绘制分布图。")
        return

    ratios_raw = df["X_times_LineVul_tokens_cover_slicing"].dropna()
    ratios = ratios_raw[ratios_raw > 0]

    filtered_out = len(ratios_raw) - len(ratios)
    if filtered_out > 0:
        print(f"[INFO] 有 {filtered_out} 个值 <=0 被过滤。")

    if len(ratios) == 0:
        print("[WARN] 没有有效样本。")
        return

    # 区间设计：0–1, 1–1.5, 1.5–2, 2–3, 3–4, ..., 9–10, >10
    bin_edges = np.array([0, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    counts_0_10, _ = np.histogram(ratios, bins=bin_edges)

    # 额外 bin：>10
    count_gt_10 = (ratios > 10).sum()

    # 整合全部计数
    counts = np.concatenate([counts_0_10, [count_gt_10]])
    total = counts.sum()
    percentages = counts / total * 100.0

    # 区间名称
    labels = [
        "0-1",
        "1-1.5",
        "1.5-2",
        "2-3",
        "3-4",
        "4-5",
        "5-6",
        "6-7",
        "7-8",
        "8-9",
        "9-10",
        ">10",
    ]

    # 保存 CSV
    cov_hist_df = pd.DataFrame({
        "bin_label": labels,
        "count": counts,
        "percentage": percentages,
    })
    cov_hist_path = os.path.join(save_dir, "X_times_CodeBERT_tokens_cover_slicing_histogram_custom_bins.csv")
    cov_hist_df.to_csv(cov_hist_path, index=False)

    # 绘图
    plt.figure(figsize=(10, 6))
    x = np.arange(len(labels))
    plt.bar(x, percentages, width=0.8)

    plt.xticks(x, labels, rotation=45, ha="right", fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlabel("X times", fontsize=25)
    plt.ylabel("Percentage of instances (%)", fontsize=25)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    cov_plot_path = os.path.join(save_dir, "X_times_CodeBERT_tokens_cover_slicing_histogram_custom_bins.png")
    plt.savefig(cov_plot_path, dpi=300)
    plt.close()

    print(f"[INFO] 图已保存到: {cov_plot_path}")
    print(f"[INFO] 数据已保存到: {cov_hist_path}")




def export_vulnerable_slicing_missed_tokens(all_files_intersection_res, output_dir):
    """
    针对 DeepWuKong 的 vulnerable slicing（xfg_file 含 'vulnerable'），
    导出每个 slicing 在 1x/2x/3x/4x/all 下：
      - slicing_tokens_number
      - missed_tokens_* / missed_ratio_*
      - file_id
      - xfg_file (具体的 DeepWuKong slicing 文件名)
    到一个 Excel 文件。
    """
    save_dir = os.path.join(output_dir, "all_files_intersection_res_CodeBERT_DeepWuKong")
    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame(all_files_intersection_res)

    if "xfg_file" not in df.columns:
        print("[WARN] Column 'xfg_file' not found, cannot filter vulnerable slicings.")
        return

    # 只保留 DeepWuKong vulnerable slicings
    mask = df["xfg_file"].astype(str).str.contains("vulnerable", na=False)
    vdf = df[mask].copy()

    if len(vdf) == 0:
        print("[WARN] No vulnerable slicings found (xfg_file not containing 'vulnerable').")
        return

    print(f"[INFO] Exporting missed-token stats for {len(vdf)} vulnerable slicings.")

    # 想导出的列（如果有缺失就自动跳过）
    desired_cols = [
        "file_id",
        "xfg_file",
        "label_slicing",
        "slicing_tokens_number",

        "selected_tokens_1x",
        "covered_tokens_1x",
        "missed_tokens_1x",
        "missed_ratio_1x",

        "selected_tokens_2x",
        "covered_tokens_2x",
        "missed_tokens_2x",
        "missed_ratio_2x",

        "selected_tokens_3x",
        "covered_tokens_3x",
        "missed_tokens_3x",
        "missed_ratio_3x",

        "selected_tokens_4x",
        "covered_tokens_4x",
        "missed_tokens_4x",
        "missed_ratio_4x",

        "selected_tokens_all",
        "covered_tokens_all",
        "missed_tokens_all",
        "missed_ratio_all",

        "selected_tokens_until_cover",
        "covered_tokens_until_cover",
        "missed_tokens_until_cover",
        "missed_ratio_until_cover",
    ]

    existing_cols = [c for c in desired_cols if c in vdf.columns]
    export_df = vdf[existing_cols].copy()

    out_path = os.path.join(save_dir, "vulnerable_slicings_missed_tokens_detail.xlsx")
    export_df.to_excel(out_path, index=False)
    print(f"[INFO] Vulnerable slicing missed-token detail saved to: {out_path}")



def analyze_jaccard_vs_coverage_distribution(all_files_intersection_res, output_dir):
    """
    统计 jaccard 的分布直方图：
      - 第 1 个柱子：0.0 <= jaccard < 0.1
      - 第 2 个柱子：0.1 <= jaccard < 0.2
      - ...
      - 第 10 个柱子：0.9 <= jaccard <= 1.0
    y 轴为每个柱子对应的样本占比（百分比）。
    结果图和数据保存到: os.path.join(output_dir, "all_files_intersection_res_CodeBERT_DeepWuKong")
    """
    save_dir = os.path.join(output_dir, "all_files_intersection_res_CodeBERT_DeepWuKong")
    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame(all_files_intersection_res)

    if "jaccard" not in df.columns:
        print("[WARN] jaccard 列不存在，无法绘制 jaccard 分布。")
        return

    jaccards_raw = df["jaccard"].dropna()

    jaccards = jaccards_raw[(jaccards_raw >= 0.0) & (jaccards_raw <= 1.0)]
    filtered_out = len(jaccards_raw) - len(jaccards)
    if filtered_out > 0:
        print(f"[INFO] 有 {filtered_out} 个 jaccard 不在 [0,1] 范围内，被过滤。")

    if len(jaccards) == 0:
        print("[WARN] 没有有效的 jaccard 样本。")
        return

    bin_edges = np.linspace(0.0, 1.0, 11)  # 0,0.1,...,1.0
    counts, _ = np.histogram(jaccards, bins=bin_edges)
    total = counts.sum()
    percentages = counts / total * 100.0

    labels = []
    for i in range(10):
        left = bin_edges[i]
        right = bin_edges[i+1]
        labels.append(f"{left:.1f}-{right:.1f}")

    jac_hist_df = pd.DataFrame({
        "bin_label": labels,
        "count": counts,
        "percentage": percentages,
    })
    jac_hist_path = os.path.join(save_dir, "jaccard_histogram_0_1_bins.csv")
    jac_hist_df.to_csv(jac_hist_path, index=False)

    # 绘图（放大字体）
    plt.figure(figsize=(8, 5))
    x = np.arange(len(labels))
    plt.bar(x, percentages, width=0.8)

    plt.xticks(x, labels, rotation=45, ha="right", fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlabel("jaccard bins", fontsize=25)
    plt.ylabel("Percentage of instances (%)", fontsize=25)
    # plt.title("Distribution of jaccard (0.0-1.0, step=0.1)", fontsize=25)

    plt.tight_layout()
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    jac_plot_path = os.path.join(save_dir, "jaccard_histogram_0_1_bins.png")
    plt.savefig(jac_plot_path, dpi=300)
    plt.close()
    print(f"[INFO] jaccard 直方图已保存到: {jac_plot_path}")
    print(f"[INFO] jaccard 直方图数据已保存到: {jac_hist_path}")





def analyze_coverage_and_jaccard_distributions(all_files_intersection_res, output_dir, bins=20):
    """
    总入口：对 all_files_intersection_res 做：
      1) X_times_LineVul_tokens_cover_slicing 分布分析
      2) jaccard vs X_times_LineVul_tokens_cover_slicing 散点分析
    """
    analyze_X_times_LineVul_tokens_cover_slicing_distribution(all_files_intersection_res, output_dir, bins=bins)
    analyze_jaccard_vs_coverage_distribution(all_files_intersection_res, output_dir)
    # 2) 六种情景下的 jaccard 分布对比
    analyze_jaccard_multi_scenarios_vulnerable_only(all_files_intersection_res, output_dir)


# ========== 主流程 ==========





def analyze_coverage_and_jaccard_distributions_vulnerable_only(all_files_intersection_res, output_dir, bins=20):
    """
    总入口：
    1. 先过滤：xfg_file 列中包含 'vulnerable' 的行
    2. 对过滤后的数据画：
       - X_times_LineVul_tokens_cover_slicing 分布图（vulnerable only）
       - X_times_LineVul_tokens_cover_slicing vs jaccard 散点图（vulnerable only）
    """
    df = pd.DataFrame(all_files_intersection_res)

    if "xfg_file" not in df.columns:
        print("[WARN] Column 'xfg_file' not found in all_files_intersection_res, skip filtering.")
        return

    # 过滤：xfg_file 路径中包含 'vulnerable'
    mask = df["xfg_file"].astype(str).str.contains("vulnerable", na=False)
    filtered_df = df[mask].copy()

    if len(filtered_df) == 0:
        print("[WARN] No rows where xfg_file contains 'vulnerable'. Nothing to analyze.")
        return

    print(f"[INFO] Filtered to {len(filtered_df)} vulnerable slicings out of {len(df)} total.")


    analyze_X_times_LineVul_tokens_cover_slicing_distribution(filtered_df, output_dir, bins=bins)
    analyze_jaccard_vs_coverage_distribution(filtered_df, output_dir)
    # 2) 六种情景下的 jaccard 分布对比
    analyze_jaccard_multi_scenarios_vulnerable_only(filtered_df.to_dict(orient="records"), output_dir)


def main():
    # 1. 初始化路径与目录
    root, c_root, output_dir, linevul_token_scores_dir = init_paths_and_dirs()

    # 2. 读取所有 C 源码文件 & 有 LineVul 结果的 file_id
    all_c_files, all_c_files_dict = get_all_c_files(root)
    available_linevul_file_ids = get_linevul_file_ids(linevul_token_scores_dir)

    # 3. XFG 目录下所有的 file_id（每个子目录对应一个函数）
    root_dir_list = sorted(os.listdir(c_root))

    all_files_intersection_res = []   # 所有 slicing 的 coverage/jaccard 结果
    pcpp_bpe_matched_res = []        # 每个函数的 PCPP/BPE 统计

    file_id_list, linevul_token_scores_dict = get_label_positive_predict_positive_id(
        linevul_token_scores_dir
    )
    # 4. 遍历每个函数 file_id
    for file_id in tqdm(root_dir_list, desc="Processing files"):
        # 只处理同时在 LineVul token 结果和 C 源码中存在的 file_id
        if (file_id not in available_linevul_file_ids) or (file_id not in all_c_files_dict):
            continue

        print("\n\nfile_id:", file_id)

        pcpp_record, slicing_metrics_list = process_one_file(
            file_id=file_id,
            all_c_files_dict=all_c_files_dict,
            c_root=c_root,
            linevul_token_scores=linevul_token_scores_dict[file_id],
            output_dir=output_dir
        )

        if pcpp_record is not None:
            pcpp_bpe_matched_res.append(pcpp_record)

        all_files_intersection_res.extend(slicing_metrics_list)

    # 5. 统一保存结果
    save_all_results(
        output_dir=output_dir,
        pcpp_bpe_matched_res=pcpp_bpe_matched_res,
        all_files_intersection_res=all_files_intersection_res
    )

    # 8. 导出 DeepWuKong vulnerable slicing 的「未选中 token」详细信息
    export_vulnerable_slicing_missed_tokens(
        all_files_intersection_res=all_files_intersection_res,
        output_dir=output_dir
    )



def analyze_missed_ratio_multi_scenarios(output_dir):
    """
    对 missed_ratio_1x、2x、3x、4x 进行区间统计并画分组柱状图。

    - 区间: 0.0-0.1, 0.1-0.2, ..., 0.9-1.0
    - 每个区间里统计:
        * 行数 (count)
        * 占该列所有有效行数的百分比 (%)
    - 输出:
        * missed_ratio_multi_scenarios.csv
        * missed_ratio_multi_scenarios.png
    """
    excel_path = "vulnerable_slicings_missed_tokens_detail.xlsx"
    save_dir = os.path.join(output_dir, "all_files_intersection_res_CodeBERT_DeepWuKong")
    excel_path = os.path.join(save_dir, excel_path)



    # 读 Excel
    df = pd.read_excel(excel_path)

    # 四种情景
    scenario_cols = {
        "missed_ratio_1x": "1× n_vul",
        "missed_ratio_2x": "2× n_vul",
        "missed_ratio_3x": "3× n_vul",
        "missed_ratio_4x": "4× n_vul",
    }

    # 10 个 bin：0.0-0.1 ... 0.9-1.0
    bin_edges = np.linspace(0.0, 1.0, 11)  # 11 个边 => 10 个 bin
    bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(10)]

    scenario_counts = {}
    scenario_percentages = {}

    for col, display_name in scenario_cols.items():
        if col not in df.columns:
            print(f"[WARN] Column '{col}' not found, skip.")
            continue

        # 去掉缺失值，并保证在 [0,1] 范围内
        vals_raw = df[col].dropna()
        vals = vals_raw[(vals_raw >= 0.0) & (vals_raw <= 1.0)]

        filtered_out = len(vals_raw) - len(vals)
        if filtered_out > 0:
            print(f"[INFO] {col}: {filtered_out} values out of [0,1] are filtered out.")

        # 统计直方图
        counts, _ = np.histogram(vals, bins=bin_edges)
        total = counts.sum()

        if total == 0:
            perc = np.zeros_like(counts, dtype=float)
        else:
            perc = counts / total * 100.0

        scenario_counts[display_name] = counts
        scenario_percentages[display_name] = perc

    if not scenario_percentages:
        print("[WARN] No valid missed_ratio_* columns, abort.")
        return

    # ---------- 保存 CSV（每个 bin 的 count 和 percent） ----------
    rows = []
    for i, label in enumerate(bin_labels):
        row = {"bin_label": label}
        for scen in scenario_percentages.keys():
            row[f"{scen}_count"] = scenario_counts[scen][i]
            row[f"{scen}_percent"] = scenario_percentages[scen][i]
        rows.append(row)

    out_df = pd.DataFrame(rows)
    csv_path = os.path.join(save_dir, "missed_ratio_multi_scenarios.csv")
    out_df.to_csv(csv_path, index=False)
    print(f"[INFO] CSV saved to: {csv_path}")

    # ---------- 画分组柱状图 ----------
    plt.figure(figsize=(12, 6))

    x = np.arange(len(bin_labels))
    scenario_names = list(scenario_percentages.keys())
    num_scenarios = len(scenario_names)
    bar_width = 0.8 / num_scenarios

    # 颜色顺序与示例图一致
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for i, (scen, color) in enumerate(zip(scenario_names, colors)):
        offset = (i - num_scenarios / 2) * bar_width + bar_width / 2
        plt.bar(
            x + offset,
            scenario_percentages[scen],
            width=bar_width,
            label=scen,
            color=color,
        )

    plt.xticks(x, bin_labels, rotation=45, ha="right", fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("Missed ratio", fontsize=22)
    plt.ylabel("Percentage (%)", fontsize=22)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.legend(title="Scenario", fontsize=16, title_fontsize=16)
    plt.tight_layout()

    png_path = os.path.join(save_dir, "missed_ratio_multi_scenarios.png")
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"[INFO] Figure saved to: {png_path}")


def main_analyze_only():
    """
    只做分析：从已经保存好的 coverage_jaccard_res.xlsx 读回数据，
    然后调用两种 analyze_* 函数画图。
    """
    # 只需要路径就够了
    root, c_root, output_dir, _ = init_paths_and_dirs()
    save_dir = os.path.join(output_dir, "all_files_intersection_res_CodeBERT_DeepWuKong")
    excel_path = os.path.join(save_dir, "coverage_jaccard_res.xlsx")

    if not os.path.exists(excel_path):
        print(f"[ERROR] {excel_path} 不存在，请先运行完整的 main() 生成该文件。")
        return

    df = pd.read_excel(excel_path)
    # 转回成原来期望的 list[dict] 结构
    all_files_intersection_res = df.to_dict(orient="records")

    # 用读取的结果做分析
    analyze_coverage_and_jaccard_distributions(
        all_files_intersection_res=all_files_intersection_res,
        output_dir=output_dir,
        bins=20
    )

    analyze_coverage_and_jaccard_distributions_vulnerable_only(
        all_files_intersection_res=all_files_intersection_res,
        output_dir=output_dir,
        bins=20
    )

    # 8. 导出 DeepWuKong vulnerable slicing 的「未选中 token」详细信息
    export_vulnerable_slicing_missed_tokens(
        all_files_intersection_res=all_files_intersection_res,
        output_dir=output_dir
    )
    analyze_missed_ratio_multi_scenarios(output_dir)


if __name__ == '__main__':
    # main()
    main_analyze_only()




