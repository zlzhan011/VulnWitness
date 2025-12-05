"""
THIS FILE IS OUTDATED
"""


#%%
import pandas as pd
import jsonlines
import tqdm
import os
print(os.listdir('../data/MSR_Linux_before_marco_preprocess/linevul_splits'))
# exit()
with jsonlines.open("../data/MSR_Linux_before_marco_preprocess/full_experiment_real_data_processed/MSR_Linux_before_marco_preprocess-full_graph.jsonlines") as reader:
    datas = [{"id": row["id"], "file_name": row["file_name"]} for row in tqdm.tqdm(reader)]
# feat_df = pd.read_json("data/MSR_Linux_before_marco_preprocess/full_experiment_real_data_processed/MSR_Linux_before_marco_preprocess-full_graph.jsonlines", lines=True)

print("datas:", datas)
print("datas len:", len(datas))
# exit()

feat_df = pd.DataFrame(data=datas)
#%%
feat_df = feat_df[~feat_df["file_name"].str.contains("_after_")]
feat_df["id"] = feat_df["file_name"].apply(lambda fn: int(fn.split("_")[0]))
# feat_df = feat_df.set_index("id")
feat_df = feat_df.rename_axis("index").reset_index(drop=True)
# feat_df

#%%
# splits_df = pd.read_csv("../data/MSR_Linux_before_marco_preprocess/linevul_splits/splits.csv")

splits_df = pd.read_csv("../data/MSR_Linux_before_marco_preprocess/linevul_splits/splits.csv").rename(columns={"index": "id"})
# splits_df

#%%
merge_df = pd.merge(feat_df, splits_df, on="id")
merge_df
# for splitname, splitdata in merge_df.groupby("label"):
#     "data/MSR_Linux_before_marco_preprocess/
merge_df = merge_df.rename(columns={"id": "index"})
#%%
merge_df.to_csv("../data/MSR_Linux_before_marco_preprocess/full_experiment_real_data_processed/vlinevul/splits.csv")
merge_df.to_csv("../data/MSR_Linux_before_marco_preprocess/full_experiment_real_data_processed/vMSR_Linux_before_marco_preprocess/splits.csv")

print(merge_df.shape)
# splitscsv_df = pd.read_csv("../data/MSR_Linux_before_marco_preprocess/full_experiment_real_data_processed/vlinevul/splits.csv")
# # splitscsv_df
#
# #%%
# m = merge_df.join(splitscsv_df, lsuffix="_ds", rsuffix="_splitscsv")
# m
#
# #%%
# import numpy as np
# assert np.all(m["split_ds"] == m["split_splitscsv"])
