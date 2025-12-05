#%%
import pandas as pd
import os
sample = False
print(os.listdir('../data/Synthetic_V3/'))
df = pd.read_csv("../data/Synthetic_V3/testv3.csv", index_col=0, nrows=5 if sample else None)
df['vul'] = df['target']
df['func_before'] = df['processed_func']
# df['func_after'] = df['processed_func']
df['project'] = 'Synthetic'
df['commit_id'] = df['index']

#%%
def get_filename(row, version):
    return "_".join([str(row.name), row["project"], str(row["commit_id"]), version, str(row["vul"])]) + ".c"
get_filename(df.iloc[0], "before")

#%%extract code

import os
code_dir = "../data/Synthetic_V3/raw_code"
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
code_dir = "../data/Synthetic_V3/raw_code"
def extract_filename(row):
    if row["vul"]:
        # versions = ["before", "after"]
        versions = ["before"]
    else:
        versions = ["before"]
    for version in versions:
        yield get_filename(row, version)
with open("../data/Synthetic_V3/files.txt", "w") as f:
    for i, row in df.iterrows():
        for filename in extract_filename(row):
            f.write(filename + "\n")
