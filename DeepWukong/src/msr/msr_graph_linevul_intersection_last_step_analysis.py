import os

import pandas as pd
import json
import Levenshtein
import networkx as nx
import numpy as np

def get_top_k_tokens_all(top_k_items, top_k_tokens):

    for item in top_k_items:
        pcpp_token_weight_avg = float(item['weight'])
        pcpp_token = item['token']
        pcpp_token_rank_id = item['rank_id']
        vote_score = (100 - int(pcpp_token_rank_id) + 1 ) * 0.01

        top_k_tokens.append({"pcpp_token_weight_avg": pcpp_token_weight_avg,
                                     "pcpp_token": pcpp_token,
                                     "pcpp_token_rank_id":pcpp_token_rank_id,
                                     "vote_score":vote_score})
    # print("top_k_items_mean_all:", top_k_items_mean_all)
    # exit()
    return top_k_tokens

def read_xfg_label(full_path):
    pkl_file = full_path.replace("XFG_LineVul_Map", "XFG").replace(".linevul_map.json", "")
    # print("\n\n\n**********pkl_file:", pkl_file)
    one_res = {}
    xfg = nx.read_gpickle(pkl_file)
    label_slicing = xfg.graph['label']
    num_nodes_slicing = len(xfg.nodes)
    num_nodes_func = 0

    one_res['xfg_file'] = pkl_file
    one_res['label_slicing'] = label_slicing
    one_res['num_nodes_slicing'] = num_nodes_slicing

    return label_slicing



def get_label_positive_predict_positive_id(output_dir):
    linevul_token_scores_dir = '/scratch/c00590656/vulnerability/LineVul/linevul/shap/saved_models_MSR/token_weight_map'
    linevul_token_scores_dir = '/scratch/c00590656/vulnerability/LineVul/linevul/shap/saved_models_MSR/token_weight_map_predict_positive_label_positive'
    file_id_list_original = os.listdir(linevul_token_scores_dir)
    file_id_list = [item.split('.')[0] for item in file_id_list_original ]
    rank_for_linevul_tokens(file_id_list_original, linevul_token_scores_dir, output_dir)

    return file_id_list





def rank_for_linevul_tokens(file_id_list, linevul_token_scores_dir, output_dir):
    top_k_tokens = []
    for file_id in file_id_list:
        file_path = os.path.join(linevul_token_scores_dir, file_id)
        df = pd.read_excel(file_path)
        # print(df)
        one_file_token_information_list = []
        for index, row in df.iterrows():
            token = row['token']
            weight = row['weight']
            one_file_token_information_list.append({'token': token, 'weight': weight})
        one_file_token_information_list = sort_tokens_and_get_rank_id(one_file_token_information_list)
        top_k_tokens = get_top_k_tokens_all(one_file_token_information_list, top_k_tokens)
    top_k_agg(top_k_tokens, output_dir, method='LineVul_token_LineVul_model_original_weight', k=100)

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


def get_top_k_instance_for_one_slicing(one_slicing_res):
    node_mapped_linevul_tokens_concat = []
    for node_index, node_corr_information in one_slicing_res.items():
        node_mapped_linevul_tokens = node_corr_information['node_mapped_linevul_tokens']
        node_mapped_linevul_tokens_concat += node_mapped_linevul_tokens

    node_mapped_linevul_tokens_concat = sort_tokens_and_get_rank_id(node_mapped_linevul_tokens_concat)

    return node_mapped_linevul_tokens_concat




def concat_rank(shap_weight_analysis_dir, method):

    filtered_mean_rank = pd.read_excel(os.path.join(shap_weight_analysis_dir, method,"filtered_mean_rank.xlsx"))
    filtered_sum_rank= pd.read_excel(os.path.join(shap_weight_analysis_dir, method,"filtered_sum_rank.xlsx"))
    filtered_median_rank= pd.read_excel(os.path.join(shap_weight_analysis_dir, method,"filtered_median_rank.xlsx"))
    filtered_count_rank= pd.read_excel(os.path.join(shap_weight_analysis_dir, method,"filtered_count_rank.xlsx"))
    agg_df_vote_score_rank= pd.read_excel(os.path.join(shap_weight_analysis_dir, method,"filtered_vote_score.xlsx"))

    filtered_mean_rank = filtered_mean_rank.sort_values(by='mean_rank', ascending=True).reset_index(drop=True)
    filtered_sum_rank = filtered_sum_rank.sort_values(by='sum_rank', ascending=True).reset_index(drop=True)
    filtered_median_rank = filtered_median_rank.sort_values(by='median_rank', ascending=True).reset_index(drop=True)
    filtered_count_rank = filtered_count_rank.sort_values(by='count_rank', ascending=True).reset_index(drop=True)
    agg_df_vote_score_rank = agg_df_vote_score_rank.sort_values(by='vote_score_rank', ascending=True).reset_index(drop=True)

    # filtered_mean_rank['rank_id'] = filtered_mean_rank['mean_rank']
    # filtered_sum_rank['rank_id'] = filtered_sum_rank['sum_rank']
    # filtered_median_rank['rank_id'] = filtered_median_rank['median_rank']
    # filtered_count_rank['rank_id'] = filtered_count_rank['count_rank']
    # agg_df_vote_score_rank['rank_id'] = agg_df_vote_score_rank['vote_score_rank']

    dfs = [
        filtered_mean_rank,
        filtered_sum_rank,
        filtered_median_rank,
        filtered_count_rank,
        agg_df_vote_score_rank
    ]
    dfs = pd.concat(dfs, axis=1)
    dfs.to_excel(os.path.join(shap_weight_analysis_dir, method,"concat_rank_res.xlsx"), index=False)

def top_k_agg(data, shap_weight_analysis_dir, method='', k=100):
    # 转换为 DataFrame
    df = pd.DataFrame(data)

    # 按 pcpp_token 进行聚合
    agg_df = df.groupby('pcpp_token')['pcpp_token_weight_avg'].agg(
        mean='mean',
        sum='sum',
        median=lambda x: np.median(x)
    ).reset_index()

    agg_df_vote_score = df.groupby('pcpp_token')['vote_score'].agg(
        sum='sum'
    ).reset_index()
    agg_df_vote_score.columns = ['pcpp_token', 'vote_score']
    token_counts = df['pcpp_token'].value_counts().reset_index()
    token_counts.columns = ['pcpp_token', 'count']
    agg_df = agg_df.merge(token_counts, on='pcpp_token', how='left')

    agg_df['mean_rank'] = agg_df['mean'].rank(method='first', ascending=False)
    agg_df['sum_rank'] = agg_df['sum'].rank(method='first', ascending=False)
    agg_df['median_rank'] = agg_df['median'].rank(method='first', ascending=False)
    agg_df['count_rank'] = agg_df['count'].rank(method='first', ascending=False)
    agg_df_vote_score['vote_score_rank'] = agg_df_vote_score['vote_score'].rank(method='first', ascending=False)
    # 筛选 mean_rank, sum_rank, median_rank 小于 30 的行
    filtered_mean_rank = agg_df[agg_df['mean_rank'] < k]
    filtered_sum_rank = agg_df[agg_df['sum_rank'] < k]
    filtered_median_rank = agg_df[agg_df['median_rank'] < k]
    filtered_count_rank = agg_df[agg_df['count_rank'] < k]
    agg_df_vote_score_rank = agg_df_vote_score[agg_df_vote_score['vote_score_rank'] < k]

    filtered_mean_rank = filtered_mean_rank[['pcpp_token', 'mean', 'mean_rank']].sort_values(by='mean_rank', ascending=True)
    filtered_sum_rank = filtered_sum_rank[['pcpp_token', 'sum', 'sum_rank']].sort_values(by='sum_rank', ascending=True)
    filtered_median_rank = filtered_median_rank[['pcpp_token', 'median', 'median_rank']].sort_values(by='median_rank', ascending=True)
    filtered_count_rank = filtered_count_rank[['pcpp_token', 'count', 'count_rank']].sort_values(by='count_rank', ascending=True)

    # print("filtered_mean_rank:")
    # print(filtered_mean_rank)
    # print("filtered_sum_rank:")
    # print(filtered_sum_rank)
    # print("filtered_median_rank:")
    # print(filtered_median_rank)
    # print("filtered_count_rank:")
    # print(filtered_count_rank)
    # print("agg_df_vote_score_rank:")
    # print(agg_df_vote_score_rank)

    if not os.path.exists(os.path.join(shap_weight_analysis_dir, method)):
        os.mkdir(os.path.join(shap_weight_analysis_dir, method))

    # 将最终结果写入 CSV 文件
    # print("shap_weight_analysis：",  shap_weight_analysis_dir)
    # print("method:", method)
    # exit()
    filtered_mean_rank.to_excel(os.path.join(shap_weight_analysis_dir, method,"filtered_mean_rank.xlsx"), index=False)
    filtered_sum_rank.to_excel(os.path.join(shap_weight_analysis_dir, method,"filtered_sum_rank.xlsx"), index=False)
    filtered_median_rank.to_excel(os.path.join(shap_weight_analysis_dir, method,"filtered_median_rank.xlsx"), index=False)
    filtered_count_rank.to_excel(os.path.join(shap_weight_analysis_dir, method,"filtered_count_rank.xlsx"), index=False)
    agg_df_vote_score_rank.to_excel(os.path.join(shap_weight_analysis_dir, method,"filtered_vote_score.xlsx"), index=False)

    concat_rank(shap_weight_analysis_dir, method)
    return {"filtered_mean_rank":filtered_mean_rank,
            "filtered_sum_rank":filtered_sum_rank,
            "filtered_median_rank":filtered_median_rank,
            "filtered_count_rank":filtered_count_rank}


if __name__ == '__main__':
    c_root = '/scratch/c00590656/vulnerability/DeepWukong/data/msr/XFG_LineVul_Map'
    output_dir = '/scratch/c00590656/vulnerability/DeepWukong/src/msr/msr_graph_linevul_intersection_last_step_analysis_res'


    shap_weight_analysis_dir = output_dir
    method = 'LineVul_token_LineVul_model_original_weight'
    concat_rank(shap_weight_analysis_dir, method)


    root_dir_list = os.listdir(c_root)
    file_id_list = get_label_positive_predict_positive_id(output_dir)
    # exit()
    ii = 0
    top_k_tokens = []
    for file_id in root_dir_list:


        if file_id in file_id_list:
            pass
        else:
            continue
        print("\n\nfile_id:", file_id)

        ii = ii + 1



        file_id_dir = os.path.join(c_root, file_id)
        one_file_id_files = []
        all_xfg_res = []

        for dirpath, dirnames, filenames in os.walk(file_id_dir):
            for filename in filenames:
                full_path = os.path.join(dirpath, filename)
                one_file_id_files.append(full_path)
                label_slicing = read_xfg_label(full_path)

                if label_slicing == 1:
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


                node_mapped_linevul_tokens_concat = get_top_k_instance_for_one_slicing(one_slicing_res)
                all_xfg_res.append(node_mapped_linevul_tokens_concat)
                top_k_tokens = get_top_k_tokens_all(node_mapped_linevul_tokens_concat, top_k_tokens)

    top_k_agg(top_k_tokens, output_dir, method='XFG_node_XFG_LABEL_1_LineVul_Token_weight', k=100)
        # # 打印或处理文件路径
        # for file_path in one_file_id_files:
        #     print(file_path)





