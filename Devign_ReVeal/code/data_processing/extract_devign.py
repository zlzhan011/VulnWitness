#%%
import pandas as pd
import os
sample = False
print(os.listdir('../data/Devign/'))
df = pd.read_csv("../data/Devign/devign_test.csv", index_col=0, nrows=5 if sample else None)
print("df_column:", df.columns)
df['vul'] = df['target']
df['func_before'] = df['processed_func']
df['func_after'] = df['processed_func']
# df

#%%
def get_filename(row, version):
    return "_".join([str(row.name), row["project"], row["commit_id"], version, str(row["vul"])]) + ".c"
get_filename(df.iloc[0], "before")

#%%extract code

import os
code_dir = "../data/Devign/raw_code"
os.makedirs(code_dir, exist_ok=True)
def extract_code(row):
    if row["vul"]:
        # versions = ["before", "after"]
        versions = ["before"]
    else:
        versions = ["before"]
    for version in versions:
        filename = get_filename(row, version)
        filepath = os.path.join(code_dir, filename)
        with open(filepath, "w") as f:
            f.write(row["func_" + version])
df.apply(extract_code, axis=1)

#%% extract filenames
code_dir = "../data/Devign/raw_code"
def extract_filename(row):
    if row["vul"]:
        # versions = ["before", "after"]
        versions = ["before"]
    else:
        versions = ["before"]
    for version in versions:
        yield get_filename(row, version)


with open("../data/Devign/files.txt", "w") as f:
    for i, row in df.iterrows():
        for filename in extract_filename(row):
            f.write(filename + "\n")
