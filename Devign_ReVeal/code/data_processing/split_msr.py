"""
THIS FILE IS OUTDATED
"""


#%%
import pandas as pd
import jsonlines
import tqdm
import os
# print(os.listdir('../data/MSR/linevul_splits'))


base_dir = '/scratch/c00590656/vulnerability/data-package/models/Devign_ReVeal/code/'
# exit()
with jsonlines.open(os.path.join(base_dir, "data/MSR/full_experiment_real_data_processed/MSR-full_graph.jsonlines")) as reader:
    datas = [{"id": row["id"], "file_name": row["file_name"]} for row in tqdm.tqdm(reader)]
# feat_df = pd.read_json("data/MSR/full_experiment_real_data_processed/MSR-full_graph.jsonlines", lines=True)
feat_df = pd.DataFrame(data=datas)
#%%
feat_df = feat_df[~feat_df["file_name"].str.contains("_after_")]
feat_df["id"] = feat_df["file_name"].apply(lambda fn: int(fn.split("_")[0]))
# feat_df = feat_df.set_index("id")
feat_df = feat_df.rename_axis("index").reset_index(drop=True)
# feat_df

#%%
# splits_df = pd.read_csv("../data/MSR/linevul_splits/splits.csv")

splits_df = pd.read_csv(os.path.join(base_dir, "data/MSR/linevul_splits/splits.csv")).rename(columns={"index": "id"})
# splits_df

#%%
merge_df = pd.merge(feat_df, splits_df, on="id")
merge_df
# for splitname, splitdata in merge_df.groupby("label"):
#     "data/MSR/
merge_df = merge_df.rename(columns={"id": "index"})

# 目标目录
directory = os.path.join(base_dir, "data/MSR/full_experiment_real_data_processed/vlinevul")

# 检查目录是否存在，如果不存在，则创建它
if not os.path.exists(directory):
    os.makedirs(directory)  # 使用 makedirs 来确保能创建多层目录结构

directory_2 = os.path.join(base_dir, 'data/MSR/full_experiment_real_data_processed/vMSR/')
if not os.path.exists(directory_2):
    os.makedirs(directory_2)  # 使用 makedirs 来确保能创建多层目录结构

#%%
merge_df.to_csv(os.path.join(directory, "splits.csv"))
merge_df.to_csv(os.path.join(directory_2,"splits.csv"))

# splitscsv_df = pd.read_csv("../data/MSR/full_experiment_real_data_processed/vlinevul/splits.csv")
# # splitscsv_df
#
# #%%
# m = merge_df.join(splitscsv_df, lsuffix="_ds", rsuffix="_splitscsv")
# m
#
# #%%
# import numpy as np
# assert np.all(m["split_ds"] == m["split_splitscsv"])
