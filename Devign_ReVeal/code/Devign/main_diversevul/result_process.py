import argparse
import pandas as pd
import os


def get_file_id(x):
    return str(x.split('_')[0])


def is_before_after(x):

    return x.split('_')[-2]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='/data1/cs_lzhan011/vulnerability/data-package/models/Devign_ReVeal/code/data/MSR/output_dir/predict_result')

    args = parser.parse_args()
    args.original_output_file = 'predict_result.xlsx'
    args.original_output_file = 'predict_result_original.xlsx'
    args.final_output_file = 'predict_result_final.xlsx'
    df = pd.read_excel(os.path.join(args.output_dir, args.original_output_file))



    df['all_inputs_ids'] = df['files_name'].apply(get_file_id)
    df['is_before_after']  = df['files_name'].apply(is_before_after)
    df.to_excel(os.path.join(args.output_dir, args.final_output_file ))



    df_after = df[df.is_before_after == 'after']
    df_after_all_inputs_ids = df_after['all_inputs_ids'].tolist()

    unpaired = []
    paired_after = []
    paired_before = []
    for i, row in df.iterrows():
        all_inputs_ids = row['all_inputs_ids']
        is_before_after = row['is_before_after']
        if all_inputs_ids not in df_after_all_inputs_ids:
            unpaired.append(row)
        else:
            if is_before_after == 'after':
                row['y_trues'] = 1
                paired_after.append(row)
            if is_before_after == 'before':
                paired_before.append(row)

    unpaired = pd.DataFrame(unpaired)
    paired_after = pd.DataFrame(paired_after)
    paired_before = pd.DataFrame(paired_before)
    paired_after = pd.concat([unpaired, paired_after])
    paired_before = pd.concat([unpaired, paired_before])

    paired_after.to_excel(os.path.join(args.output_dir, 'predict_result_after_final_original.xlsx' ))
    paired_before.to_excel(os.path.join(args.output_dir, 'predict_result_before_final_original.xlsx'))



