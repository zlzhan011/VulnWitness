
import os
import pandas as pd




if __name__ == '__main__':
    c_dir = '/data1/cs_lzhan011/vulnerability/data-package/models/Devign_ReVeal/code/data/MSR/full_experiment_real_data_processed/vMSR'
    file_path = os.path.join(c_dir, 'splits.csv')
    df = pd.read_csv(file_path)
    paired_index = []
    for i, row in df.iterrows():
        file_name = row['file_name']
        index = row['index']
        if '_1.c' in file_name:
            paired_index.append(index)


    # pared_dir = '/data1/cs_lzhan011/vulnerability/data-package/models/Devign_ReVeal/code/data/MSR/parsed/tmp'
    #
    # paired_index = [file  for file in os.listdir(pared_dir) if 'after' in file]
    # paired_index = [file.split('_')[0] for file in paired_index]

    paired_example = []
    for i, row in df.iterrows():
        file_name = row['file_name']
        index = row['index']
        if index in paired_index:
            paired_example.append(row)

    paired_example = pd.DataFrame(paired_example)
    paired_example.to_csv(os.path.join(c_dir, 'paired_example.csv'))


