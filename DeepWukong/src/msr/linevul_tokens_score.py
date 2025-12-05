import os
import pickle
import numpy as np
import pandas as pd


def read_linevul_token_scores(instance_id, linevul_token_scores_dir):
    file_list = os.listdir(linevul_token_scores_dir)
    file_name = str(instance_id)+".pkl.xlsx"

    if file_name in file_list:
        file_path = os.path.join(linevul_token_scores_dir, file_name)

        try:
            df = pd.read_excel(file_path)
        except:
            print("file_path:", file_path)
            df = []
    else:
        return []
    return  df



