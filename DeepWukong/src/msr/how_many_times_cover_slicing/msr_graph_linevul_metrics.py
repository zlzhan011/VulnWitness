import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import List, Tuple


# ===================== 工具函数：加载与预处理 =====================

def load_data(
    base_dir: str,
    missed_filename: str = "vulnerable_slicings_missed_tokens_detail.xlsx",
    pred_filename: str = "test_processed_func_prediction_20250418.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    从指定目录中加载：
      - vulnerable_slicings_missed_tokens_detail.xlsx
      - test_processed_func_prediction_20250418.csv
    """
    missed_detail_path = os.path.join(base_dir, missed_filename)
    pred_path = os.path.join(base_dir, pred_filename)

    df_missed = pd.read_excel(missed_detail_path)
    df_pred = pd.read_csv(pred_path)

    print("[INFO] missed detail columns:", df_missed.columns.tolist())
    print("[INFO] prediction columns:", df_pred.columns.tolist())
    print(f"[INFO] Loaded missed detail from: {missed_detail_path}")
    print(f"[INFO] Loaded prediction from:    {pred_path}")

    return df_missed, df_pred


def extract_file_id_from_path(path: str) -> int:
    """
    从路径/文件名中抽取 file_id（假设是 basename 的第一个 '_' 之前的部分）
    例如：'12345_vulnerable_xxx.gpickle' -> 12345
    """
    if pd.isna(path):
        return np.nan
    fname = os.path.basename(str(path))
    first = fname.split('_')[0]
    try:
        return int(first)
    except ValueError:
        return np.nan


def attach_file_id_to_missed_detail(
    df_missed: pd.DataFrame,
    path_col: str = "xfg_file"
) -> pd.DataFrame:
    """
    在 missed detail DataFrame 上根据路径列（如 'xfg_file'）解析出 file_id 列。
    """
    if path_col not in df_missed.columns:
        raise KeyError(f"[ERROR] 指定的路径列 '{path_col}' 不存在于 missed detail 中。")

    df_missed = df_missed.copy()
    # df_missed["file_id"] = df_missed[path_col].apply(extract_file_id_from_path)
    before_drop = len(df_missed)
    df_missed = df_missed.dropna(subset=["file_id"])
    df_missed["file_id"] = df_missed["file_id"].astype(int)

    print(f"[INFO] Attached file_id to missed detail. "
          f"Dropped {before_drop - len(df_missed)} rows with invalid file_id.")
    return df_missed


def attach_file_id_to_predictions(df_pred: pd.DataFrame) -> pd.DataFrame:
    """
    在预测结果 DataFrame 上基于 all_inputs_ids 添加 file_id 列。
    默认假设 all_inputs_ids 对应 MSR 的 index。
    """
    if "all_inputs_ids" not in df_pred.columns:
        raise KeyError("[ERROR] 预测结果中缺少 'all_inputs_ids' 列。")

    df_pred = df_pred.copy()
    df_pred["file_id"] = df_pred["all_inputs_ids"].astype(int)

    # 保留需要的列
    needed_cols = ["file_id", "y_trues", "LineVul_original_Predictions"]
    missing = [c for c in needed_cols if c not in df_pred.columns]
    if missing:
        raise KeyError(f"[ERROR] 预测结果中缺少列: {missing}")

    df_pred_small = df_pred[needed_cols]
    print(f"[INFO] Predictions shrunk to columns {needed_cols}.")
    return df_pred_small


def merge_missed_and_pred(
    df_missed_with_id: pd.DataFrame,
    df_pred_with_id: pd.DataFrame
) -> pd.DataFrame:
    """
    根据 file_id 将 missed detail 与预测结果合并。
    每一行代表一个 DeepWukong vulnerable slicing 及其对应的 LineVul 预测。
    """
    df_merged = df_missed_with_id.merge(df_pred_with_id, on="file_id", how="inner")
    print(f"[INFO] Merged df shape: {df_merged.shape}")
    return df_merged


# ===================== 分桶 + 指标计算 =====================

def compute_metrics_per_missed_ratio(
    df_merged: pd.DataFrame,
    ratio_col: str,
    ratio_name: str,
) -> pd.DataFrame:
    """
    对某个 missed_ratio_* 列：
      - 按 0.0–0.1, 0.1–0.2, ..., 0.9–1.0 分成 10 个 level
      - 在每个 level 上，计算 precision / recall / f1

    返回 DataFrame：
      [ratio_type, ratio_col, level, bin_left, bin_right, count, precision, recall, f1]
    """
    if ratio_col not in df_merged.columns:
        print(f"[WARN] 列 {ratio_col} 不存在，跳过。")
        return pd.DataFrame()

    sub = df_merged.dropna(subset=[ratio_col]).copy()
    if sub.empty:
        print(f"[WARN] 列 {ratio_col} 全部为 NaN，跳过。")
        return pd.DataFrame()

    # 构造 0.0–1.0 的 10 个区间
    bin_edges = np.linspace(0.0, 1.0, 11)  # [0.0, 0.1, ..., 1.0]
    bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(10)]

    # 使用 right=True, include_lowest=True：
    # -> [0.0,0.1], (0.1,0.2], ..., (0.9,1.0]
    sub[f"{ratio_col}_level"] = pd.cut(
        sub[ratio_col],
        bins=bin_edges,
        labels=bin_labels,
        include_lowest=True,
        right=True
    )

    records = []
    for i, label in enumerate(bin_labels):
        bin_df = sub[sub[f"{ratio_col}_level"] == label]
        n = len(bin_df)
        if n == 0:
            continue

        y_true = bin_df["y_trues"].values
        y_pred = bin_df["LineVul_original_Predictions"].values

        # level 内如果只有一个类别会导致除零，zero_division=0 避免报错
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        records.append({
            "ratio_type": ratio_name,   # 例如 "missed_ratio_1x"
            "ratio_col": ratio_col,     # 实际列名
            "level": label,             # "0.0-0.1" 等
            "bin_left": bin_edges[i],
            "bin_right": bin_edges[i+1],
            "count": n,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        })

    result_df = pd.DataFrame(records)
    print(f"[INFO] {ratio_col} -> computed metrics for {len(result_df)} non-empty levels.")
    return result_df


def analyze_all_missed_ratios(df_merged: pd.DataFrame) -> pd.DataFrame:
    """
    对 df_merged 中的 missed_ratio_1x/2x/3x/4x 逐个做分桶与指标计算，
    并汇总到一个 DataFrame 返回。
    """
    ratio_cols: List[str] = [
        "missed_ratio_1x",
        "missed_ratio_2x",
        "missed_ratio_3x",
        "missed_ratio_4x",
    ]

    all_results = []
    for col in ratio_cols:
        res_df = compute_metrics_per_missed_ratio(
            df_merged=df_merged,
            ratio_col=col,
            ratio_name=col
        )
        if not res_df.empty:
            all_results.append(res_df)

    if all_results:
        final_res = pd.concat(all_results, ignore_index=True)
    else:
        final_res = pd.DataFrame()
        print("[WARN] 没有任何 missed_ratio_* 相关结果。")

    return final_res


# ===================== 结果保存 =====================

def save_results(
    final_res: pd.DataFrame,
    base_dir: str,
    out_filename: str = "missed_ratio_level_metrics.xlsx"
) -> str:
    """
    将最终结果 DataFrame 保存为 Excel，返回保存路径。
    """
    out_path = os.path.join(base_dir, out_filename)
    final_res.to_excel(out_path, index=False)
    print(f"[INFO] 结果已保存到: {out_path}")
    return out_path


# ===================== 主入口 =====================

def main():
    """
    主流程：
      1. 设置路径和列名
      2. 加载数据
      3. 解析并对齐 file_id
      4. 合并 missed detail 与预测结果
      5. 对 missed_ratio_1x/2x/3x/4x 逐个做分桶分析
      6. 保存最终结果到一个 Excel
    """

    # 1) 路径配置（根据你的实际路径调整 base_dir）
    base_dir = (
        "/scratch/c00590656/vulnerability/DeepWukong/src/msr/"
        "msr_graph_linevul_intersection_last_step_analysis_res/all_files_intersection_res"
    )

    missed_filename = "vulnerable_slicings_missed_tokens_detail.xlsx"
    pred_filename = "test_processed_func_prediction_20250418.csv"

    # DeepWukong slicing 路径的列名（里面包含 file_id 信息）
    # 如果你的列名不是 'xfg_file'，这里改成实际列名
    PATH_COL = "xfg_file"

    # 2) 加载数据
    df_missed, df_pred = load_data(
        base_dir=base_dir,
        missed_filename=missed_filename,
        pred_filename=pred_filename,
    )

    # 3) 在 missed detail 中解析 file_id
    df_missed_with_id = attach_file_id_to_missed_detail(df_missed, path_col=PATH_COL)

    # 4) 在预测结果中附加 file_id
    df_pred_with_id = attach_file_id_to_predictions(df_pred)

    # 5) 合并两个表
    df_merged = merge_missed_and_pred(df_missed_with_id, df_pred_with_id)

    # 6) 对 missed_ratio_1x/2x/3x/4x 做分桶 + P/R/F1 计算
    final_res = analyze_all_missed_ratios(df_merged)

    if final_res.empty:
        print("[WARN] 最终结果为空，请检查数据或列名是否正确。")
        return

    print("[INFO] Final results preview:")
    print(final_res.head(20))

    # 7) 保存结果
    save_results(final_res, base_dir=base_dir)


if __name__ == "__main__":
    main()
