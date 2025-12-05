import os

import pandas as pd
import json
import Levenshtein



def calcul_Lenvenshtein_ration(joined_tokens, node_code):


    # # 示例字符串
    # joined_tokens = "structsvc_rdma_op_ctxt*ctxt"
    # node_code = "struct svc_rdma_op_ctxt * ctxt"

    # 去除所有空格
    joined_tokens_clean = joined_tokens.replace(" ", "")
    node_code_clean = node_code.replace(" ", "")

    # 计算 Levenshtein 距离
    lev_distance = Levenshtein.distance(joined_tokens_clean, node_code_clean)

    # 计算相似度（normalized ratio：1 表示完全一样）
    similarity_ratio = Levenshtein.ratio(joined_tokens_clean, node_code_clean)

    # print(f"Levenshtein 距离: {lev_distance}")
    # print(f"相似度（0~1）：{similarity_ratio:.4f}")

    return similarity_ratio


def msr_graph_linevul_intersection_last_step(linevul_token_weight, xfg_nodes_add_code_information, pkl_file):
    # print("linevul_token_weight:\n", linevul_token_weight)
    xfg_nodes_add_code_information_add_linevul_token_weight = {}
    for k, v in xfg_nodes_add_code_information.items():
        # print("k, v:\n", k, v)
        location_updated = v['location_updated']
        code_updated = v['code_updated']

        if "&&&&&" in location_updated:
            # print("location_updated:", location_updated)
            location_start, location_end = location_updated.split("#")[0], location_updated.split("#")[1]
            # print("location_start:", location_start)
            # print("location_end:", location_end)
            location_start_list = location_start.split("&&&&&")
            location_end_list = location_end.split("&&&&&")
            node_code = code_updated.split("&&&&&")[0]

            # print("location_start_list:", location_start_list)
            # print("location_end_list:", location_end_list)

            location_start_list = [int(item) for item in location_start_list]
            location_end_list = [int(item) for item in location_end_list]


        else:

            if "#" in location_updated:
                location_start, location_end = int(location_updated.split("#")[0]), int(location_updated.split("#")[1])
                location_start_list = [location_start]
                location_end_list = [location_end]
            else:
                location_start_list = []

        all_potential_mapped_linevul_tokens = []
        for i in range(len(location_start_list)):
            one_location_start = location_start_list[i]
            one_location_end = location_end_list[i]
            mapped_linevul_tokens = []
            for index, row in linevul_token_weight.iterrows():
                start_offset = row['start_offset']
                end_offset = row['end_offset']
                token = row['token']
                weight = row['weight']
                if (start_offset>=one_location_start and start_offset<=one_location_end) and \
                        (end_offset>=one_location_start and end_offset<=one_location_end):
                    mapped_linevul_tokens.append({"token":token,
                                                  "start_offset":start_offset,
                                                  "end_offset":end_offset,
                                                  "weight":weight})

            mapped_linevul_tokens = sorted(mapped_linevul_tokens, key=lambda x: x['start_offset'])
            all_potential_mapped_linevul_tokens.append(mapped_linevul_tokens)


        if len(all_potential_mapped_linevul_tokens) == 0:
            mapped_linevul_tokens = []
        elif len(all_potential_mapped_linevul_tokens) == 1:
            mapped_linevul_tokens =   all_potential_mapped_linevul_tokens[0]
        else:

            similarity_ratio_list = []

            for one_mapped_linevul_tokens in all_potential_mapped_linevul_tokens:
                joined_tokens = ''.join(item['token'] for item in one_mapped_linevul_tokens)

                similarity_ratio = calcul_Lenvenshtein_ration(joined_tokens, node_code)
                similarity_ratio_list.append(similarity_ratio)

            max_similarity_ratio = max(similarity_ratio_list)
            max_index = similarity_ratio_list.index(max_similarity_ratio)
            mapped_linevul_tokens = all_potential_mapped_linevul_tokens[max_index]

        # print("mapped_linevul_tokens:\n", mapped_linevul_tokens)
        if len(mapped_linevul_tokens) == 0:
            node_sum_weight, node_average_weight = 0, 0
        else:
            # 求和
            node_sum_weight = sum(item['weight'] for item in mapped_linevul_tokens)

            # 平均值
            node_average_weight = node_sum_weight / len(mapped_linevul_tokens)

        v['node_mapped_linevul_tokens'] = mapped_linevul_tokens
        v['node_average_weight'] = node_average_weight
        v['node_sum_weight'] = node_sum_weight

        xfg_nodes_add_code_information_add_linevul_token_weight[k] = v

    output_file = pkl_file.replace('XFG', 'XFG_LineVul_Map')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    output_file = output_file+".linevul_map.json"

    with open(output_file, 'w') as f:
        f.write(str(xfg_nodes_add_code_information_add_linevul_token_weight))

    return xfg_nodes_add_code_information_add_linevul_token_weight








# if __name__ == '__main__':
#     c_root = '/scratch/c00590656/vulnerability/DeepWukong/data/msr/linevul_graph_intersection'
#
#     linevul_token_weight =  pd.read_excel(os.path.join(c_root, '181346_linevul_token_weight.xlsx'))
#
#     # 读取文件
#     with open(os.path.join(c_root, '181346_xfg_nodes_add_code_information.json'), 'r') as f:
#         xfg_nodes_add_code_information = json.load(f)
#
#     pkl_file = '/scratch/c00590656/vulnerability/DeepWukong/data/msr/XFG/181346/array/32.xfg.pkl'
#     msr_graph_linevul_intersection_last_step(linevul_token_weight, xfg_nodes_add_code_information, pkl_file)
#
#




