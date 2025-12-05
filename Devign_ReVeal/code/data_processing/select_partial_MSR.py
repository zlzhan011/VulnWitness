

import pandas as pd
import os
sample = False
print(os.listdir('../data/MSR/'))
df = pd.read_csv("../data/MSR/MSR_data_cleaned.csv", index_col=0, nrows=5 if sample else None)

df_partial = df.head(1000)
df_partial.to_csv('../data/MSR/MSR_data_cleaned_partial.csv')