import pickle

import matplotlib.pyplot as plt  # 如果不再用画图，可以删掉
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


# ====================== 新增：构造 LaTeX 段落所需的结构 ======================

def build_segments_for_latex_bak(original_code, slicing_tokens, linevul_tokens_sorted):
    """
    根据：
      - vulnerable slicing subgraph 映射到的 tokens（slicing_tokens）
      - LineVul SHAP 排序后的 tokens（linevul_tokens_sorted）
    生成：
      - segments: 每个 segment 对应一个 [start_offset, end_offset) 区间，并记录：
          * 是否属于 slicing（is_slicing）
          * shap_rank / shap_group
          * start_anchor / end_anchor（Seg{i}S / Seg{i}E）
      - vulnerable_slicing_size: slicing token 的数量
      - minted_code: 已经插好 \mymark 的代码字符串，可直接放进 minted 环境中
    """
    seg_map = {}

    def get_seg(start, end):
        key = (start, end)
        if key not in seg_map:
            seg_map[key] = {
                "start": start,
                "end": end,
                "is_slicing": False,
                "shap_rank": None,    # 该位置最好的(最小)rank
                "shap_group": None,   # 0,1,2,3
            }
        return seg_map[key]

    # ---- 1) vulnerable slicing tokens ----
    slicing_locs = set()
    for t in slicing_tokens:
        s = int(t["start_offset"])
        e = int(t["end_offset"])
        slicing_locs.add((s, e))
        seg = get_seg(s, e)
        seg["is_slicing"] = True

    vulnerable_slicing_size = len(slicing_locs)

    # ---- 2) LineVul SHAP 排序后的 tokens ----
    for rank, tok in enumerate(linevul_tokens_sorted, start=1):
        s = int(tok["start_offset"])
        e = int(tok["end_offset"])
        seg = get_seg(s, e)
        # 同一位置可能出现多次，取 rank 最小的那次
        if seg["shap_rank"] is None or rank < seg["shap_rank"]:
            seg["shap_rank"] = rank

    # ---- 3) 根据 vulnerable_slicing_size 把 shap_rank 映射到分组 ----
    for seg in seg_map.values():
        rank = seg["shap_rank"]
        if rank is None or vulnerable_slicing_size == 0:
            seg["shap_group"] = None
        else:
            # 0: 0-1x, 1:1-2x, 2:2-3x, 3: >=3x
            bin_index = (rank - 1) // vulnerable_slicing_size
            if bin_index <= 0:
                group = 0
            elif bin_index == 1:
                group = 1
            elif bin_index == 2:
                group = 2
            else:
                group = 3
            seg["shap_group"] = group

    # ---- 4) 排序 + 生成 anchor 名字 ----
    segments = sorted(seg_map.values(), key=lambda s: (s["start"], s["end"]))
    for i, seg in enumerate(segments, start=1):
        seg["start_anchor"] = f"Seg{i}S"
        seg["end_anchor"]   = f"Seg{i}E"

    # ---- 5) 用 segments 把 original_code 包一层 \mymark，生成 minted_code ----
    pieces = []
    last_pos = 0

    for seg in segments:
        s = seg["start"]
        e = seg["end"]
        start_anchor = seg["start_anchor"]
        end_anchor   = seg["end_anchor"]

        # 补上前面没有标记覆盖的原始代码
        if s > last_pos:
            pieces.append(original_code[last_pos:s])

        seg_text = original_code[s:e]

        # 注意这里用 (*@ ... @*)，因为 minted 设置了 escapeinside=@@
        pieces.append(
            f"(*@\\mymark{{{start_anchor}}}@*)"
            f"{seg_text}"
            f"(*@\\mymark{{{end_anchor}}}@*)"
        )

        last_pos = e

    # 末尾剩余代码
    if last_pos < len(original_code):
        pieces.append(original_code[last_pos:])

    minted_code = "".join(pieces)

    # 返回多一个 minted_code，可以直接写进 .tex 的 minted 环境里
    return segments, vulnerable_slicing_size


from typing import List, Dict, Tuple

# ===================== 工具：转义单个源代码字符 =====================

def escape_code_char(c: str) -> str:
    """
    把单个源代码字符转成适合 LaTeX 正文里显示的形式。
    注意：这里不处理空格和换行，它们原样保留，用于之后做缩进和换行。
    """
    if c == '\\':
        return r'\textbackslash{}'
    elif c == '_':
        return r'\_'
    elif c == '&':
        return r'\&'
    elif c == '%':
        return r'\%'
    elif c == '$':
        return r'\$'
    elif c == '#':
        return r'\#'
    elif c == '{':
        return r'\{'
    elif c == '}':
        return r'\}'
    elif c == '~':
        return r'\textasciitilde{}'
    elif c == '^':
        return r'\^{}'
    elif c == '<':
        return r'\textless{}'
    elif c == '>':
        return r'\textgreater{}'
    else:
        return c



from typing import List, Tuple, Union, Set

StyleType = Union[str, Tuple[str, ...]]  # 第3个元素可以是 'red' 或 ('red','uline')

def split_overlapping_segments(
    raw_segments: List[Tuple[int, int, str]]
) -> List[Tuple[int, int, StyleType]]:
    """
    输入:
        raw_segments: [(start, end, style_str), ...]
            - style_str 比如 'black', 'red', 'uline', 'purple' 等
            - 区间视为 [start, end) 半开区间
    输出:
        merged_segments: [(start, end, style_or_tuple), ...]，满足:
            1. 所有区间不重叠，按 start, end 排序；
            2. 若某位置同时被颜色段和 uline 段覆盖，
               第3个元素是一个 tuple，例如 ('purple', 'uline')；
            3. 若只有一种样式，则第3个元素仍然是一个字符串，例如 'black'。
    """

    # 1) 把所有 start/end 事件打平
    events = []  # (pos, is_start, style)
    for s, e, style in raw_segments:
        if s >= e:
            continue
        events.append((s, 1, style))   # 1 表示 start
        events.append((e, -1, style))  # -1 表示 end

    if not events:
        return []

    # 按位置排序；同一位置时，先处理 end(-1) 再处理 start(1)，避免零长度片段
    events.sort(key=lambda x: (x[0], x[1]))

    merged: List[Tuple[int, int, StyleType]] = []
    active: Set[str] = set()  # 当前区间内激活的样式集合
    prev_pos = None
    i = 0
    n = len(events)

    # 2) 扫描线：在相邻事件位置之间生成区间
    while i < n:
        pos = events[i][0]

        # 在 [prev_pos, pos) 之间，如果有激活样式，就输出一个区间
        if prev_pos is not None and prev_pos < pos and active:
            # 把 active 的样式集合转换成字符串或 tuple
            styles = sorted(active)
            if len(styles) == 1:
                style_out: StyleType = styles[0]      # 只有一种样式 -> 字符串
            else:
                style_out = tuple(styles)              # 多种样式 -> tuple
            merged.append((prev_pos, pos, style_out))

        # 处理当前位置上所有事件（可能有多个 start/end）
        while i < n and events[i][0] == pos:
            _, flag, style = events[i]
            if flag == 1:        # start
                active.add(style)
            else:                # end
                active.discard(style)
            i += 1

        prev_pos = pos

    # 不需要处理尾部，因为 [prev_pos, +∞) 不再有事件；若你希望保留尾巴可在此扩展

    return merged


def convert_styles_to_tuple(segments):
    """
    输入: [(start, end, style), ...]
    style 可以是:
      - 'red'
      - 'uline'
      - ('red','uline')

    输出: 每个 style 都变成 tuple:
      - 'red' -> ('red',)
      - ('red','uline') -> ('red','uline')
    """
    new_segments = []
    for s, e, style in segments:
        if isinstance(style, tuple):
            new_style = style
        else:
            # 字符串转换成 ('string',)
            new_style = (style,)
        new_segments.append((s, e, new_style))
    return new_segments


# ===================== 1. 从 linevul_tokens_sorted 构造 segments =====================

def build_color_segments(
    linevul_tokens_sorted: List[Dict],
slicing_tokens
) -> List[Tuple[int, int, str]]:
    """
    输入:
      - linevul_tokens_sorted: 每个元素包含
          * 'start_offset'
          * 'end_offset'
          * 'shap_group_label'
    输出:
      - segments: [(start, end, color_name), ...]，按 start 升序，并合并同色重叠区间。
    """
    group_to_color = {
        '>=3x': 'black',
        '0-1x': 'red',
        '1-2x': 'green',
        '2-3x': 'purple',
    }

    raw_segments: List[Tuple[int, int, str]] = []
    for tok in linevul_tokens_sorted:
        label = tok.get('shap_group_label')
        color = group_to_color.get(label)
        if color is None:
            continue  # 不需要着色的分组直接跳过

        start = int(tok['start_offset'])
        end = int(tok['end_offset'])
        if start >= end:
            continue
        raw_segments.append((start, end, color))

    if not raw_segments:
        return []


    for item in slicing_tokens:
        start = int(item['start_offset'])
        end = int(item['end_offset']) + 1
        raw_segments.append((start, end, 'uline'))

    # 按 start 排序
    raw_segments.sort(key=lambda x: x[0])

    raw_segments = split_overlapping_segments(raw_segments)
    raw_segments = convert_styles_to_tuple(raw_segments)
    return raw_segments


# ===================== 2. 先按 segments 把 original_code 切块 + 上色 =====================

def split_code_by_segments(
    original_code: str,
    segments: List[Tuple[int, int, Tuple[str, ...]]],
) :
    """
    根据 segments 把 original_code 切成一块一块：
      返回 [(text, styles_or_None), ...]
      styles_or_None 为 ('red',) / ('red','uline') / ('uline',) / None
    假定 segments 已经按 start 升序且不互相交叉。
    """
    chunks = []
    pos = 0
    n = len(original_code)

    for start, end, styles in segments:
        start = max(0, min(start, n))
        end = max(0, min(end, n))
        if start > end:
            start, end = end, start

        if pos < start:
            # 无样式的普通代码块
            chunks.append((original_code[pos:start], None))
        # 有样式的代码块
        chunks.append((original_code[start:end], styles))
        pos = end

    if pos < n:
        chunks.append((original_code[pos:], None))

    return chunks


def chunks_to_latex_with_colors(
    chunks
) -> str:
    parts: List[str] = []

    for text, styles in chunks:
        # 1) 先转义
        esc_chars: List[str] = []
        for c in text:
            if c == ' ' or c == '\n':
                esc_chars.append(c)  # 空格和换行原样保留
            else:
                esc_chars.append(escape_code_char(c))
        esc_text = ''.join(esc_chars)

        # 2) 解析样式：颜色 + 是否下划线
        color = None
        uline = False

        if styles is not None:
            for s in styles:
                if s == "uline":
                    uline = True
                else:
                    color = s

        # 3) 按顺序包 LaTeX 命令：先颜色，再外围 uline
        styled = esc_text
        if color is not None:
            styled = rf'\textcolor{{{color}}}{{{styled}}}'
        if uline:
            styled = rf'\uline{{{styled}}}'

        parts.append(styled)

    return ''.join(parts)


# ===================== 3. 按行加 \hspace* 与 \\ =====================

def add_indent_and_linebreaks(
    colored_escaped_code: str,
    indent_unit_pt: float = 2.0,
) -> str:
    """
    输入：
      - colored_escaped_code:  已经完成转义 + \textcolor 包裹，
        但还保留原始空格和 '\n'
    输出：
      - 多行 LaTeX，每行前面用 \hspace*{N pt} 表示缩进，行尾加 '\\'
    """
    lines = colored_escaped_code.split('\n')
    latex_lines: List[str] = []

    for line in lines:
        # 统计行首空格数
        i = 0
        while i < len(line) and line[i] == ' ':
            i += 1
        indent_spaces = i
        indent_cmd = ''
        if indent_spaces > 0:
            indent_cmd = rf'\hspace*{{{indent_spaces * indent_unit_pt}pt}}'

        content = line[indent_spaces:]
        latex_lines.append(indent_cmd + content + r'\\')

    return '\n'.join(latex_lines)


# ===================== 4. 组合上述步骤，得到 {\small\ttfamily ...} 内部内容 =====================

def code_body_to_latex_with_segments(
    original_code: str,
    linevul_tokens_sorted: List[Dict],
    slicing_tokens,
  indent_unit_pt: float = 2.0,
) -> str:
    """
    整体流程：
      1) 从 linevul_tokens_sorted 构造 segments；
      2) 按 segments 把 original_code 切块；
      3) 每块内部转义并按需包 \textcolor；
      4) 再按照行首空格加 \hspace*，行尾加 '\\'。
    返回的就是可以直接放进 {\small\ttfamily ... } 的内容。
    """
    segments = build_color_segments(linevul_tokens_sorted, slicing_tokens)
    chunks = split_code_by_segments(original_code, segments)
    colored_escaped = chunks_to_latex_with_colors(chunks)
    body = add_indent_and_linebreaks(colored_escaped, indent_unit_pt=indent_unit_pt)
    return body


# ===================== 5. 拼完整的 LIPIcs LaTeX 文档 =====================

def generate_full_lipics_latex(
    original_code: str,
    linevul_tokens_sorted: List[Dict],
slicing_tokens
) -> str:

    body = code_body_to_latex_with_segments(original_code, linevul_tokens_sorted, slicing_tokens)

    full_tex = (
        r"\makeatletter" "\n"
        r"\def\input@path{{/scratch/c00590656/vulnerability/DeepWukong/src/msr/how_many_times_cover_slicing/}}" "\n"
        r"\makeatother" "\n"
        r"\documentclass[a4paper,UKenglish,cleveref, autoref, thm-restate,anonymous]{lipics-v2021}" "\n"
        r"\usepackage[T1]{fontenc}" "\n"
        r"\usepackage{xcolor}" "\n"
        r"\usepackage[normalem]{ulem}" "\n"
        r"\usepackage{textcomp} % 提供 \textasciitilde" "\n"
        "\n"
        r"\begin{document}" "\n"
        "\n"
        r"{\small\ttfamily" "\n"
        r"\noindent" "\n"
        f"{body}\n"
        r"}" "\n"
        r"\end{document}" "\n"
    )
    return full_tex







import json  # 你后面已经有 import json，如果已写就不用再加

def save_token_info_json(
    tex_path: str,
    file_id: str,
    original_code: str,
    slicing_tokens: list,
    linevul_tokens: list,
    segments: list,
    vulnerable_slicing_size: int,
) -> str:
    """
    将每个 file 对应的：
      - code 文本 original_code
      - LineVul token 及其分组信息（0-1x/1-2x/2-3x/≥3x）
      - slicing 的 token
    保存到和 LaTeX 文件同一目录下的一个 .json 文件中。
    """

    # 1) 先做一个 (start,end) -> shap_group 的映射，方便给 LineVul token 附上分组
    span2group = {}
    for seg in segments:
        s = int(seg["start"])
        e = int(seg["end"])
        span2group[(s, e)] = seg.get("shap_group")

    def group_label(g):
        if g is None:
            return None
        if g == 0:
            return "0-1x"
        if g == 1:
            return "1-2x"
        if g == 2:
            return "2-3x"
        return ">=3x"

    # 2) 处理 LineVul token：附上 shap_group 信息
    linevul_tokens_out = []
    for tok in linevul_tokens:
        s = int(tok["start_offset"])
        e = int(tok["end_offset"])
        g = span2group.get((s, e))
        linevul_tokens_out.append({
            "token": tok.get("token"),
            "weight": tok.get("weight"),
            "start_offset": s,
            "end_offset": e,
            "rank_id": tok.get("rank_id"),
            "shap_group": g,
            "shap_group_label": group_label(g),
        })

    # 3) 处理 slicing token（vulnerable slicing subgraph 映射到的 token）
    slicing_tokens_out = []
    for t in slicing_tokens:
        slicing_tokens_out.append({
            "token": t.get("token"),
            "start_offset": int(t["start_offset"]),
            "end_offset": int(t["end_offset"]),
        })

    # 4) 组织 JSON 数据
    data = {
        "file_id": file_id,
        "vulnerable_slicing_size": vulnerable_slicing_size,
        "original_code": original_code,
        "slicing_tokens": slicing_tokens_out,
        "linevul_tokens": linevul_tokens_out,
    }

    # 5) JSON 路径：与 tex_path 同目录同前缀
    json_path = os.path.splitext(tex_path)[0] + ".json"
    json_dir = os.path.dirname(json_path)
    os.makedirs(json_dir, exist_ok=True)

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    print("[OK] JSON saved:", json_path)
    return json_path







# ====================== 你原来的函数，基本保留 ======================

def draw_code_with_ulines_scaled(text, green_ranges, black_ranges, full_path, flaw_line_index_dict, file_id):
    # 这个函数如果你以后不用画 matplotlib，可以删掉
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
    uline_offset = line_spacing * 0.3  # 红线在下
    doubleline_gap = line_spacing * 0.2  # 黑线再下方一点

    for line in lines:
        x = 0.0
        for char in line:
            ax.text(x, y, char, fontsize=font_size, va='center', family='monospace')

            offset_idx = 0  # 用于错开多条下划线的距离

            if char_index in green_indices:
                y_offset = -uline_offset - offset_idx * 0.005
                ax.plot([x, x + char_width], [y + y_offset, y + y_offset], color='green', linewidth=1)
                offset_idx += 1

            if char_index in black_indices_black:
                y_offset = -uline_offset - offset_idx * 0.005
                ax.plot([x, x + char_width], [y + y_offset, y + y_offset], color='black', linewidth=1)
                offset_idx += 1

            x += char_spacing
            char_index += 1

        char_index += 1  # for '\n'
        y -= line_spacing

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    plt.tight_layout()

    full_path_jpg = full_path.replace("XFG", "XFG_Intersection_Visual_X_Times") + ".jpg"
    full_path_pkl = full_path.replace("XFG", "XFG_Intersection_Visual_X_Times") + ".pkl"
    full_path_json = full_path.replace("XFG", "XFG_Intersection_Visual_X_Times") + ".json"
    # 只取目录部分（不包含文件名）
    dir_path = os.path.dirname(full_path_jpg)
    # 创建目录（如果已存在不会报错）
    os.makedirs(dir_path, exist_ok=True)

    print("full_path_jpg:", full_path_jpg)
    plt.savefig(full_path_jpg)
    intersection_res = {"text": text, "green_ranges": green_ranges, "black_ranges": black_ranges}

    import json
    with open(full_path_pkl, "wb") as f:
        pickle.dump(intersection_res, f)

    with open(full_path_json, "w") as f:
        json.dump(intersection_res, f)

    plt.show()


def rank_for_linevul_tokens(file_id_list, linevul_token_scores_dir):
    linevul_token_scores_dict = {}
    for file_id in file_id_list:
        file_path = os.path.join(linevul_token_scores_dir, file_id)
        df = pd.read_excel(file_path)
        one_file_token_information_list = []
        for index, row in df.iterrows():
            token = row['token']
            weight = row['weight']
            one_file_token_information_list.append(
                {
                    'token': token,
                    'weight': weight,
                    "start_offset": row['start_offset'],
                    "end_offset": row['end_offset'],
                }
            )
        one_file_token_information_list = sort_tokens_and_get_rank_id(one_file_token_information_list)
        linevul_token_scores_dict[file_id] = one_file_token_information_list
    return linevul_token_scores_dict


def sort_tokens_and_get_rank_id(token_data):
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
        item.pop("original_order", None)

    return sorted_tokens


def get_label_positive_predict_positive_id():
    linevul_token_scores_dir = '/scratch/c00590656/vulnerability/LineVul/linevul/shap/saved_models_MSR/token_weight_map_predict_positive_label_positive'
    file_id_list_original = os.listdir(linevul_token_scores_dir)
    file_id_list = [item.split('.')[0] for item in file_id_list_original]
    linevul_token_scores_dict = rank_for_linevul_tokens(file_id_list_original, linevul_token_scores_dir)
    return file_id_list, linevul_token_scores_dict


def read_xfg_label(full_path):
    pkl_file = full_path.replace("XFG_LineVul_Map", "XFG").replace(".linevul_map.json", "")

    if os.path.getsize(pkl_file) == 0:
        print("\n\n\n**********error pkl_file:", pkl_file)
        return 2   # 2 meaning error

    xfg = nx.read_gpickle(pkl_file)
    label_slicing = xfg.graph['label']
    num_nodes_slicing = len(xfg.nodes)
    num_nodes_func = 0

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


def read_flaw_lines_index():
    c_root = '/scratch/c00590656/vulnerability/LineVul/data/big-vul_dataset'
    c_output = '/scratch/c00590656/vulnerability/DeepWukong/data/msr/source-code/'

    flaw_line_index_dict = {}
    for file in ['val.csv', 'test.csv', 'train.csv']:
        file_path = os.path.join(c_root, file)
        file_df = pd.read_csv(file_path)
        print(file_df.columns.values)
        for ii, row in file_df.iterrows():
            func_before = row['func_before']
            func_after = row['func_after']
            CVE_ID = row['CVE ID']
            CWE_ID = row['CWE ID']
            target = row['target']
            index = row['index']
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

                    flaw_line_index_dict[index] = {
                        "index": index,
                        "flaw_line_index": flaw_line_index,
                        "flaw_line_index_max": flaw_line_index_max,
                        "target": target,
                        "project": project,
                        "func_before": func_before,
                    }

    return flaw_line_index_dict


# 新增的小工具：根据 slicing 数量选择 top_k（如果你还想用之前的画图逻辑的话）
def select_top_tokens_by_slicing(linevul_tokens, slicing_tokens, multiple=2):
    slicing_locs = {(int(t["start_offset"]), int(t["end_offset"])) for t in slicing_tokens}
    n_slicing = len(slicing_locs)

    if n_slicing == 0:
        print("Warning: slicing_tokens is empty, no LineVul top tokens will be selected.")
        return []

    top_k = multiple * n_slicing
    print(f"n_slicing = {n_slicing}, select top_k = {top_k} LineVul tokens")

    return linevul_tokens[:top_k]


import ast
import json
from tqdm import tqdm


def process_slicing_file(
    file_id: str,
    full_path: str,
    original_code: str,
    linevul_token_scores_dict: dict,
) -> None:
    """
    处理单个 slicing 结果文件：
      1. 读取并过滤 label，只保留 vulnerable slicing；
      2. 解析 slicing JSON，抽取 vulnerable slicing subgraph 映射到的 tokens 及其坐标；
      3. 读取对应的 LineVul SHAP 排序后的 token 列表；
      4. 调用 build_segments_for_latex 构造 segments；
      5. 调用 generate_latex_for_file 生成 minted + tikz 的 LaTeX 代码。
    """
    # 1) 读取 slicing 图的 label，过滤非 vulnerable 的情况
    label_slicing = read_xfg_label(full_path)
    # if not (label_slicing == 1 or (label_slicing == 0 and "vulnerable" in full_path)):
    #     # 非 vulnerable slicing，直接跳过
    #     return

    if not "vulnerable" in full_path:
        return []

    print(f"  [slicing] process file: {full_path}")

    # 2) 解析 slicing JSON 内容
    with open(full_path, "r") as f:
        content = f.read()
    content = content.replace("nan", "None")
    one_slicing_res = ast.literal_eval(content)

    # --- 构造 slicing sub-graph 对应的 token 列表 ---
    node_mapped_linevul_tokens_concat = []
    for node_index, node_corr_information in one_slicing_res.items():
        # 视你的数据结构而定，这里沿用你原来的逻辑跳过 node_index == 1
        if node_index == 1:
            continue
        node_mapped_linevul_tokens = node_corr_information['node_mapped_linevul_tokens']
        node_mapped_linevul_tokens_concat = node_mapped_linevul_tokens_concat + node_mapped_linevul_tokens
    return node_mapped_linevul_tokens_concat







import subprocess
import os

import subprocess
import os

def compile_latex_to_image(tex_path, output_format="png", clean_aux=True):
    """
    给定一个 .tex 文件：
      1. 用 pdflatex 编译（带 -shell-escape，支持 minted）
      2. 尝试用 ImageMagick (magick/convert) 把 PDF 转成 PNG/JPG
      3. 若因安全策略等失败，再尝试 pdftoppm / gs 等工具
    """
    tex_path = os.path.abspath(tex_path)
    tex_dir = os.path.dirname(tex_path)
    tex_filename = os.path.basename(tex_path)
    base, _ = os.path.splitext(tex_filename)

    pdf_path = os.path.join(tex_dir, base + ".pdf")
    if output_format is None:
        img_path = None
    else:
        img_path = os.path.join(tex_dir, base + f".{output_format}")

    # ---------- Step 1: pdflatex 编译 ----------
    cmd_pdflatex = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-shell-escape",          # ⚠ minted 必需
        "-output-directory", tex_dir,
        tex_filename,
    ]
    print("[pdflatex]", " ".join(cmd_pdflatex))
    result = subprocess.run(cmd_pdflatex, cwd=tex_dir)

    if result.returncode != 0 or not os.path.exists(pdf_path):
        print(f"[ERROR] pdflatex failed or PDF not generated for {tex_path}")
        log_path = os.path.join(tex_dir, base + ".log")
        if os.path.exists(log_path):
            print(f"  See log file: {log_path}")
        return None

    # 若不需要图片，直接返回
    if output_format is None:
        return None

    # ---------- Step 2: 尝试 ImageMagick ----------
    success = False
    tried_tools = []

    for tool in [["magick", "convert"], ["convert"]]:
        tried_tools.append(" ".join(tool))
        try:
            cmd_convert = tool + [
                "-density", "200",     # 清晰度
                pdf_path,
                "-quality", "100",
                img_path,
            ]
            print("[convert]", " ".join(cmd_convert))
            result2 = subprocess.run(cmd_convert)
            if result2.returncode == 0 and os.path.exists(img_path):
                success = True
                break
        except FileNotFoundError:
            continue  # 该工具不存在，试下一个

    # ---------- Step 3: 若 ImageMagick 因安全策略等失败，尝试 pdftoppm ----------
    if not success and output_format == "png":
        # 尝试 pdftoppm (poppler-utils)
        tried_tools.append("pdftoppm")
        try:
            # 会生成 base-1.png, base-2.png ...
            out_prefix = os.path.join(tex_dir, base)
            cmd_pdftoppm = [
                "pdftoppm",
                "-png",
                "-r", "200",          # 分辨率
                pdf_path,
                out_prefix,
            ]
            print("[pdftoppm]", " ".join(cmd_pdftoppm))
            result3 = subprocess.run(cmd_pdftoppm)
            first_png = out_prefix + "-1.png"
            if result3.returncode == 0 and os.path.exists(first_png):
                # 重命名成我们统一使用的 img_path
                os.replace(first_png, img_path)
                success = True
        except FileNotFoundError:
            pass

    # ---------- Step 4: 若 pdftoppm 也不可用，尝试 Ghostscript ----------
    if not success and output_format == "png":
        tried_tools.append("gs")
        try:
            cmd_gs = [
                "gs",
                "-dSAFER",
                "-dBATCH",
                "-dNOPAUSE",
                "-sDEVICE=png16m",
                "-r200",
                f"-sOutputFile={img_path}",
                pdf_path,
            ]
            print("[gs]", " ".join(cmd_gs))
            result4 = subprocess.run(cmd_gs)
            if result4.returncode == 0 and os.path.exists(img_path):
                success = True
        except FileNotFoundError:
            pass

    # ---------- 检查最终结果 ----------
    if not success or not os.path.exists(img_path):
        print(f"[ERROR] Image not generated for {pdf_path}")
        print("  Tried tools:", tried_tools)
        img_path = None

    # ---------- 清理中间文件 ----------
    if clean_aux:
        for ext in [".aux", ".log", ".out"]:
            p = os.path.join(tex_dir, base + ext)
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

    if img_path is not None:
        print("[OK] Image saved:", img_path)
    return img_path




def process_one_file_id(
    file_id: str,
    c_root: str,
    all_c_files_dict: dict,
    linevul_token_scores_dict: dict,
) -> None:
    """
    处理单个 file_id 的完整流程：
      1. 找到该 file_id 对应的 C 源码文件（func_before）并读入 original_code；
      2. 遍历 c_root/file_id 目录下的所有 slicing 结果文件；
      3. 对每个 slicing 结果调用 process_slicing_file，生成对应的 LaTeX。
    """
    print(f"\n\n[file_id] {file_id}")

    # 1) 读取 C 源码内容
    if file_id not in all_c_files_dict:
        print("  [warn] file_id not in all_c_files_dict:", file_id)
        return

    c_file_path = all_c_files_dict[file_id]
    with open(c_file_path, "r") as f:
        original_code = f.read()

    # 2) 遍历该 file_id 目录下的所有 slicing 文件
    file_id_dir = os.path.join(c_root, file_id)
    if not os.path.isdir(file_id_dir):
        print("  [warn] slicing dir not found:", file_id_dir)
        return

    for dirpath, dirnames, filenames in os.walk(file_id_dir):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            # 这里不再在主循环里写复杂逻辑，全部交给 process_slicing_file
            node_mapped_linevul_tokens_concat = process_slicing_file(
                file_id=file_id,
                full_path=full_path,
                original_code=original_code,
                linevul_token_scores_dict=linevul_token_scores_dict,
            )
            if not node_mapped_linevul_tokens_concat==[] and node_mapped_linevul_tokens_concat is not None:
                slicing_linevul_scores_x_times, slicing_num = process_linevul_groups(
                    node_mapped_linevul_tokens_concat,
                    linevul_token_scores_dict[file_id+".pkl.xlsx"],
                )
                full_write_path = full_path.replace("XFG_LineVul_Map", "XFG_Intersection_Visual_X_Times_LineVul_Map")
                full_write_dir = os.path.dirname(full_write_path)
                os.makedirs(full_write_dir, exist_ok=True)
                with open(full_write_path, "w") as f:
                    json.dump({"file_id":file_id,
                                    "original_code":original_code,
                                    "LineVul_Tokens": linevul_token_scores_dict[file_id+".pkl.xlsx"],
                                    "slicing_node_mapped_linevul_tokens_concat":node_mapped_linevul_tokens_concat,
                                    "slicing_linevul_scores_x_times":slicing_linevul_scores_x_times,
                             }, f)

from typing import List, Dict, Any, Tuple

def dedup_dict_list(dict_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    对由 dict 组成的列表去重（以 key-value 对为整体作为判定标准）
    """
    seen = set()
    result = []
    for d in dict_list:
        # 用排序后的 (key, value) 元组作为 hash key
        key = tuple(sorted(d.items()))
        if key not in seen:
            seen.add(key)
            result.append(d)
    return result


def process_linevul_groups(
    slicing_node_mapped_linevul_tokens_concat: List[Any],
    linevul_token_scores: List[Dict[str, Any]],
) -> Tuple[List[Any], List[Dict[str, Any]], int]:
    """
    1. 两个列表都去重；
    2. 计算去重后 node_mapped_linevul_tokens_concat 的长度，记为 slicing_contain_linevul_tokens_num；
    3. 按 weight 从大到小排序 linevul_token_scores；
    4. 给不同区间的 token 打上 shap_group_label:
         0 ~ 1x*n  -> '0-1x'
         1x*n ~ 2x*n -> '1-2x'
         2x*n ~ 3x*n -> '2-3x'
         3x*n ~ 4x*n -> '3-4x'
         >= 4x*n     -> '>=4x'
    注意：假设每个 dict 里都有 key 'weight'。
    """

    # --- 2) 计算 slicing_contain_linevul_tokens_num ---
    slicing_contain_linevul_tokens_num = len(slicing_node_mapped_linevul_tokens_concat)

    linevul_token_scores.sort(key=lambda x: x.get("weight", 0), reverse=True)

    n = slicing_contain_linevul_tokens_num

    # --- 4) 按区间添加 shap_group_label ---
    for idx, token_info in enumerate(linevul_token_scores):
        if idx < 1 * n:
            grp = "0-1x"
        elif idx < 2 * n:
            grp = "1-2x"
        elif idx < 3 * n:
            grp = "2-3x"
        elif idx < 4 * n:
            grp = "3-4x"
        else:
            # 这里我按逻辑写成 >=4x，如果你坚持用 ">=3x" 就把下面改成 ">=3x"
            grp = ">=4x"

        token_info["shap_group_label"] = grp

    return linevul_token_scores, slicing_contain_linevul_tokens_num



def main():
    """
    总入口函数：
      1. 初始化路径与数据（源码路径、XFG 路径、SHAP 路径等）；
      2. 读取 func_before 源码索引；
      3. 读取 flaw_line_index（目前只在旧的 matplotlib 可视化中用到，如不需要可以删）；
      4. 读取 LineVul SHAP 排序后的 tokens；
      5. 遍历所有 file_id，调用 process_one_file_id。
    """
    # 1) 基础路径配置
    root = "/scratch/c00590656/vulnerability/DeepWukong/data/msr/source-code/"
    c_root = "/scratch/c00590656/vulnerability/DeepWukong/src/data/msr/XFG_LineVul_Map"
    output_dir = (
        "/scratch/c00590656/vulnerability/DeepWukong/src/msr/"
        "msr_graph_linevul_intersection_last_step_analysis_res"
    )
    os.makedirs(output_dir, exist_ok=True)

    # 2) 读取所有 C 源码文件列表以及 file_id -> 源码路径 的映射
    all_c_files, all_c_files_dict = get_all_c_files(root)

    # 3) flaw_line_index（如果只用 LaTeX 不画 matplotlib，可以视情况删掉）
    flaw_line_index_dict = read_flaw_lines_index()
    print("Loaded flaw_line_index for", len(flaw_line_index_dict), "vulnerable functions")

    # 4) 读取 LineVul SHAP 排序后的 tokens
    file_id_list, linevul_token_scores_dict = get_label_positive_predict_positive_id()
    print("Loaded LineVul SHAP tokens for", len(file_id_list), "files")

    # 5) 遍历 XFG_LineVul_Map 目录下的所有 file_id，逐个处理
    root_dir_list = os.listdir(c_root)

    for file_id in tqdm(root_dir_list, desc="Processing file_ids"):
        process_one_file_id(
            file_id=file_id,
            c_root=c_root,
            all_c_files_dict=all_c_files_dict,
            linevul_token_scores_dict=linevul_token_scores_dict,
        )


def build_verbatim_code_with_highlight(original_code, segments):
    """
    使用 fvextra 的 Verbatim 环境 + commandchars=\\@#
    把不同类型的 segment 包成：
      - slicing token:      \\HLred@...#
      - 0-1x  group(0):     \\CGreen@...#
      - 1-2x  group(1):     \\CYellow@...#
      - 2-3x  group(2):     \\CRed@...#
      - >=3x group(3):      \\CPurple@...#

    其中:
      - is_slicing == True 的 segment 一律认为是 slicing subgraph，对应红色下划线 (\\HLred)
      - 其它 segment 按 shap_group 上色
    """
    # 确保有 start/end 字段且按顺序
    segments_sorted = sorted(segments, key=lambda s: (s["start"], s["end"]))
    out = []
    cur = 0

    for seg in segments_sorted:
        s = seg["start"]
        e = seg["end"]

        # 先补上中间未高亮的原始代码
        if s > cur:
            out.append(original_code[cur:s])

        seg_text = original_code[s:e]

        # 选择对应的 LaTeX 宏
        if seg.get("is_slicing", False):
            macro = "HLred"   # slicing: 红色下划线
        else:
            g = seg.get("shap_group", None)
            if g == 0:
                macro = "CGreen"   # 0-1x
            elif g == 1:
                macro = "CYellow"  # 1-2x
            elif g == 2:
                macro = "CRed"     # 2-3x
            elif g == 3:
                macro = "CPurple"  # >=3x
            else:
                macro = None

        if macro is not None:
            # 使用 Verbatim 的 commandchars=\\@#
            # 形式: \宏@内容#
            out.append(f"\\{macro}@{seg_text}#")
        else:
            out.append(seg_text)

        cur = e

    # 收尾
    if cur < len(original_code):
        out.append(original_code[cur:])

    return "".join(out)




import math
import pandas as pd  # 你本来就有引入的话可复用，没有也可以不引

def _normalize_token(value):
    """把 token 统一转成字符串；NaN / None 都视为 ''。"""
    if value is None:
        return ""
    # 处理 numpy.nan / float('nan')
    if isinstance(value, float) and math.isnan(value):
        return ""
    # 处理 pandas 的 NA / NaN
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def _normalize_token(value):
    """把 token 统一转成字符串；None/NaN 当成空串。"""
    import math
    import pandas as pd

    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def merge_slicing_tokens_by_gap(slicing_tokens, max_gap=2):
    """
    对 slicing_tokens 做合并：
      1. 按 start_offset 升序排序；
      2. 相邻两个片段 current 和 tok，计算
            gap_end   = max(current.start_offset, tok.start_offset)
            gap_start = min(current.end_offset,   tok.end_offset)
            gap       = abs(gap_end - gap_start)
         如果 gap <= max_gap，则合并：
            start_offset = min(...)
            end_offset   = max(...)
            token        = token 拼接
         否则结束 current，开启新的片段；
      3. 合并后的片段继续和后面的比较，直到不能再合并。

    返回：合并后的 slicing_tokens 列表
    """
    if not slicing_tokens:
        return []

    # 1) 先按 start_offset 排序
    tokens_sorted = sorted(slicing_tokens, key=lambda x: x["start_offset"])

    merged = []
    current = dict(tokens_sorted[0])
    current["token"] = _normalize_token(current.get("token", ""))

    for tok in tokens_sorted[1:]:
        tok_token = _normalize_token(tok.get("token", ""))

        cur_start = current["start_offset"]
        cur_end   = current["end_offset"]
        tok_start = tok["start_offset"]
        tok_end   = tok["end_offset"]

        gap_end = max(cur_start, tok_start)
        gap_start = min(cur_end, tok_end)
        gap = abs(gap_end - gap_start)

        if gap <= max_gap:
            # 合并：扩展区间 + token 拼接
            current["start_offset"] = min(cur_start, tok_start)
            current["end_offset"]   = max(cur_end, tok_end)

            if current["token"] and tok_token:
                current["token"] = current["token"] + tok_token
            else:
                current["token"] = current["token"] + tok_token
        else:
            merged.append(current)
            current = dict(tok)
            current["token"] = tok_token

    merged.append(current)
    return merged



def merge_tokens_by_offset_and_group(linevul_tokens):
    if not linevul_tokens:
        return []

    # 1) 先按 start_offset 排序
    sorted_tokens = sorted(linevul_tokens, key=lambda x: x["start_offset"])

    merged = []
    current = dict(sorted_tokens[0])

    # 确保 current 里的 token 已经是干净的
    current["token"] = _normalize_token(current.get("token", ""))

    for tok in sorted_tokens[1:]:
        tok_token = _normalize_token(tok.get("token", ""))

        if tok.get("shap_group_label") == current.get("shap_group_label"):
            # 同 group 合并
            gap_end = max(current["start_offset"], tok["start_offset"])
            gap_start  = min(current["end_offset"], tok["end_offset"])
            gap = abs(gap_end - gap_start)
            if gap<= 2:
                current["start_offset"] = min(current["start_offset"], tok["start_offset"])
                current["end_offset"] = max(current["end_offset"], tok["end_offset"])

            # 这里根据你需要决定是否加空格：
            # 比如想直接拼： "abc" + "def"
            # 或者带空格："abc" + " " + "def"
            current["token"] = current["token"] + tok_token
        else:
            merged.append(current)
            current = dict(tok)
            current["token"] = tok_token

    merged.append(current)
    return merged


def save_tex_and_image(json_path: str, tex: str, output_format: str = "png"):
    """
    根据 json_path 的位置，将 tex 内容写入：
        <json_path>.tex
    并可选调用 compile_latex_to_image 生成 PNG/JPG。

    参数:
        json_path: JSON 文件的绝对路径
        tex:       要写入的完整 LaTeX 代码
        output_format: "png" / "jpg" / None（不生成图片）
    返回:
        tex_path, image_path (如未生成图片则为 None)
    """
    json_path = os.path.abspath(json_path)

    # 1. tex 文件路径（沿用你的习惯：xxx.json → xxx.json.tex）
    tex_path = json_path.replace("XFG_Intersection_LaTeX", "XFG_Intersection_Visual_X_Times_LineVul_Map") + ".tex"
    tex_dir = os.path.dirname(tex_path)
    os.makedirs(tex_dir, exist_ok=True)

    # 2. 写 .tex 文件
    with open(tex_path, "w") as f:
        f.write(tex)

    print(f"[OK] LaTeX saved: {tex_path}")

    # 3. 编译成图片
    image_path = None
    if output_format is not None:
        image_path = compile_latex_to_image(
            tex_path,
            output_format=output_format,
            clean_aux=True,
        )

    return tex_path, image_path



def generate_lipics_latex_from_json(json_path: str, output_format: str = "png"):
    """
    从 JSON 中读取:
      - original_code
      - slicing_tokens（vulnerable slicing subgraph 映射到的 tokens）
      - linevul_tokens（LineVul + SHAP 权重信息）

    然后：
      1. 计算 vulnerable_slicing_size；
      2. 按 SHAP 排序 linevul_tokens；
      3. 用 build_segments_for_latex 构造 segments（包含 is_slicing, shap_group）；
      4. 对相邻且“样式一致”的 segments 做合并 (merge_adjacent_segments)；
      5. 用 build_verbatim_code_with_highlight 生成带 \HLxxx / \Cxxx 命令的代码串；
      6. 用 lipics-v2021 模板 + Verbatim 环境 生成 LaTeX；
      7. 如需可调用 compile_latex_to_image 编译并转成 PNG。
    """
    import json
    json_path = os.path.abspath(json_path)
    print(f"\n[JSON] processing {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)

    file_id = data.get("file_id", "unknown")
    original_code = data["original_code"]
    slicing_tokens = data.get("slicing_node_mapped_linevul_tokens_concat", [])
    linevul_tokens = data.get("slicing_linevul_scores_x_times", []) # slicing_linevul_scores_x_times 是linevul_tokens含有 x time组别的版本

    linevul_tokens = merge_tokens_by_offset_and_group(linevul_tokens)
    slicing_tokens = merge_slicing_tokens_by_gap(slicing_tokens, max_gap=2)

    # ---------- 1) 计算 vulnerable_slicing_size ----------
    slicing_locs = {
        (int(t["start_offset"]), int(t["end_offset"]))
        for t in slicing_tokens
    }
    vulnerable_slicing_size = len(slicing_locs)
    print(f"[JSON] file_id={file_id}, vulnerable_slicing_size={vulnerable_slicing_size}")

    # ---------- 2) 按 SHAP 排序 LineVul tokens ----------
    if not linevul_tokens:
        print("[WARN] linevul_tokens is empty in JSON:", json_path)
        return None

    tex = generate_full_lipics_latex(original_code, linevul_tokens, slicing_tokens)

    tex_path, img_path = save_tex_and_image(
        json_path=json_path,
        tex=tex,
        output_format=output_format
    )


    return tex_path


def batch_generate_lipics_from_json(root_dir: str, output_format: str = "png"):
    root_dir = os.path.abspath(root_dir)
    print(f"[BATCH] scan json under: {root_dir}")

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.lower().endswith(".json"):
                continue

            json_path = os.path.join(dirpath, fname)

            generate_lipics_latex_from_json(json_path, output_format=output_format)

            # try:
            #     generate_lipics_latex_from_json(json_path, output_format=output_format)
            # except Exception as e:
            #     print(f"[ERROR] failed on {json_path}: {e}")


import os
import shutil
from typing import Optional


def collect_png_and_pdf(
        root_dir: str,
        png_out_dir: str,
        pdf_out_dir: str,
        tex_out_dir:str,
        verbose: bool = True,
) -> None:
    """
    遍历 root_dir 下所有子目录，收集：
        - 所有 .png 文件到 png_out_dir
        - 所有 .pdf 文件到 pdf_out_dir
    若目标目录不存在会自动创建。

    参数:
        root_dir:     要遍历的根目录
        png_out_dir:  所有 png 汇总的目录
        pdf_out_dir:  所有 pdf 汇总的目录
        verbose:      是否打印复制日志
    """
    root_dir = os.path.abspath(root_dir)
    png_out_dir = os.path.abspath(png_out_dir)
    pdf_out_dir = os.path.abspath(pdf_out_dir)
    tex_out_dir = os.path.abspath(tex_out_dir)
    # 自动创建输出目录
    os.makedirs(png_out_dir, exist_ok=True)
    os.makedirs(pdf_out_dir, exist_ok=True)
    os.makedirs(tex_out_dir, exist_ok=True)
    if verbose:
        print(f"[INFO] Scanning: {root_dir}")

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            src_path = os.path.join(dirpath, fname)
            file_id = src_path.split("/")[-3]
            # ---- PNG ----
            if fname.lower().endswith(".png"):
                dst_path = os.path.join(png_out_dir, file_id+"_"+fname)
                shutil.copy2(src_path, dst_path)
                if verbose:
                    print(f"[PNG] {src_path} -> {dst_path}")

            # ---- PDF ----
            elif fname.lower().endswith(".pdf"):
                dst_path = os.path.join(pdf_out_dir, file_id + "_" + fname)
                shutil.copy2(src_path, dst_path)
                if verbose:
                    print(f"[PDF] {src_path} -> {dst_path}")

            elif fname.lower().endswith(".tex"):
                dst_path = os.path.join(tex_out_dir, file_id + "_" + fname)
                shutil.copy2(src_path, dst_path)
                if verbose:
                    print(f"[tex] {src_path} -> {dst_path}")

    if verbose:
        print("[DONE] PNG 和 PDF 文件已完成汇总。")



if __name__ == "__main__":
    main()
    # json_root = "/scratch/c00590656/vulnerability/DeepWukong/src/data/msr/XFG_Intersection_Visual_X_Times_LineVul_Map"
    # batch_generate_lipics_from_json(json_root)

    # collect_png_and_pdf(
    #     root_dir=json_root,
    #     png_out_dir=json_root+ "_png",
    #     pdf_out_dir=json_root+ "_pdf",
    #     tex_out_dir = json_root+ "_tex"
    # )


