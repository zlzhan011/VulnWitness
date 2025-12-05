import argparse
import pandas as pd
import os


def get_file_id(x):
    return str(x.split('_')[0])


def is_before_after(x):

    return x.split('_')[-2]


def process_prediction_file(args, is_before_after, get_file_id):
    args.original_output_file = 'predict_result_test_dataset_evaluate.xlsx'
    args.after_output_file = 'predict_result_after_function_evaluate.xlsx'
    df = pd.read_excel(os.path.join(args.output_dir, args.original_output_file))
    df_after = pd.read_excel(os.path.join(args.output_dir, args.after_output_file))

    df['all_inputs_ids'] = df['files_name'].apply(get_file_id)
    df['is_before_after'] = df['files_name'].apply(is_before_after)
    df_after['all_inputs_ids'] = df_after['files_name'].apply(get_file_id)
    df_after['is_before_after'] = df_after['files_name'].apply(is_before_after)

    if 'y_trues' in df.columns.values:
        target_column = 'y_trues'
    else:
        target_column = 'targets'
    unpaired = df[df[target_column] == 0]
    paired_before = df[df[target_column] == 1]
    paired_before_ids = paired_before['all_inputs_ids'].tolist()

    paired_after = []
    for index, row in df_after.iterrows():
        all_inputs_ids = row['all_inputs_ids']
        is_before_after = row['is_before_after']
        if all_inputs_ids in paired_before_ids:
            row[target_column] = 1
            paired_after.append(row)

    paired_after_1 = pd.DataFrame(paired_after)
    paired_before = pd.DataFrame(paired_before)

    after_before_paired = pd.merge(paired_after_1, paired_before, how='left', left_on='all_inputs_ids',
                                   right_on='all_inputs_ids')
    paired_after_2 = pd.concat([unpaired, paired_after_1])
    paired_before = pd.concat([unpaired, paired_before])

    paired_after_2.to_excel(os.path.join(args.output_dir, 'val_after_prediction.xlsx'))
    paired_before.to_excel(os.path.join(args.output_dir, 'val_before_prediction.xlsx'))
    after_before_paired.to_excel(os.path.join(args.output_dir, 'after_before_paired.xlsx'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--output_dir', default='/data1/cs_lzhan011/vulnerability/data-package/models/Devign_ReVeal/code/data/MSR/output_dir/predict_result/reveal/add_after_to_before')
    # parser.add_argument('--output_dir', default='/data1/cs_lzhan011/vulnerability/data-package/models/Devign_ReVeal/code/data/MSR/output_dir/predict_result/Devign/add_after_to_before')
    parser.add_argument('--output_dir_models', default='/data1/cs_lzhan011/vulnerability/data-package/models/Devign_ReVeal/code/data/DiverseVul/output_dir/predict_result/')

    args = parser.parse_args()

    models = ['devign', 'ggnn']

    for model in models:
        args.output_dir_model = os.path.join(args.output_dir_models, model) # ggnn devign
        for bool_item in [True, False]:
            args.training_add_after_into_before = bool_item
            args.output_dir = os.path.join(args.output_dir_model, 'training_add_after_into_before_' + str(args.training_add_after_into_before))

            process_prediction_file(args, is_before_after, get_file_id)






