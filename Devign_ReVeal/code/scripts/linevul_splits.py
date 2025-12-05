#%%
import os.path

import pandas as pd


base_dir = '/scratch/c00590656/vulnerability/data-package/models/Devign_ReVeal/code/'
test_df = pd.read_csv(os.path.join(base_dir, "data/MSR/linevul_splits/test.csv"))
test_df["split"] = "test"
train_df = pd.read_csv(os.path.join(base_dir,"data/MSR/linevul_splits/train.csv"))
train_df["split"] = "train"
valid_df = pd.read_csv(os.path.join(base_dir,"data/MSR/linevul_splits/valid.csv"))
valid_df["split"] = "valid"

#%%
merge_df = pd.concat((train_df[["index", "split"]], valid_df[["index", "split"]], test_df[["index", "split"]]), ignore_index=True)
merge_df = merge_df.set_index("index").sort_index()
merge_df

#%%
merge_df.value_counts("split", normalize=True)

#%%
merge_df.to_csv(os.path.join(base_dir,"data/MSR/linevul_splits/splits.csv"))
merge_df.to_csv(os.path.join(base_dir,"data/MSR/full_experiment_real_data_processed/vlinevd/splits.csv"))

#%%
split_df = pd.read_csv(os.path.join(base_dir,"data/MSR/full_experiment_real_data_processed/vlinevd/splits.csv"))
split_df

# %%
