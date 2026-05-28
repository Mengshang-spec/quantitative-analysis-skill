# -*- coding: utf-8 -*-
"""
基础探索性分析：整体统计、相关性、分组聚合。
"""

import pandas as pd
import numpy as np


def summary_stats(df):
    """
    输出每列的统计信息：计数、均值、标准差、分位数等。
    返回 describe 的 DataFrame。
    """
    stats = df.describe(include="all")
    print("===== 数据整体摘要 =====")
    print(stats.round(2).to_string())
    return stats


def missing_report(df):
    """
    生成缺失值报告：每列缺失数、缺失比例。
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"missing_count": missing, "missing_pct(%)": missing_pct})
    report = report[report["missing_count"] > 0].sort_values("missing_count", ascending=False)
    if len(report) == 0:
        print("无缺失值！")
    else:
        print("===== 缺失值报告 =====")
        print(report.to_string())
    return report


def correlation_matrix(df, method="pearson", top_n=20):
    """
    数值列之间的相关系数矩阵。

    参数:
        method: "pearson" / "spearman" / "kendall"
        top_n: 只展示前 top_n 列（避免矩阵过大）

    返回:
        相关系数矩阵 DataFrame
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] == 0:
        print("[WARN] 没有数值列，无法计算相关系数")
        return pd.DataFrame()

    # 若列过多，按方差选出 top_n
    if numeric_df.shape[1] > top_n:
        variances = numeric_df.var().sort_values(ascending=False)
        selected = variances.head(top_n).index
        numeric_df = numeric_df[selected]
        print(f"[INFO] 列数较多，只展示方差最大的前 {top_n} 列")

    corr = numeric_df.corr(method=method)
    print(f"===== 相关系数矩阵 ({method}) =====")
    print(corr.round(2).to_string())
    return corr


def column_type_report(df):
    """
    分析并报告各列的类型，帮助用户决定哪些列用于分类、哪些用于数值分析。
    返回分类列、数值列、日期列的列表。
    """
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()

    print("===== 列类型报告 =====")
    print(f"数值列 ({len(numeric)}): {numeric}")
    print(f"分类列 ({len(categorical)}): {categorical}")
    print(f"日期列 ({len(datetime_cols)}): {datetime_cols}")

    return numeric, categorical, datetime_cols


def auto_detect_analysis_columns(df):
    """
    自动检测适合分析的列：
    - category_col: 第一个分类列（object/category）
    - second_cat_col: 第二个分类列（如果有）
    - value_col: 第一个数值列（如果有）
    - target_col: 最后一个数值列（用于预测）

    返回 dict
    """
    numeric_cols, cat_cols, _ = column_type_report(df)

    result = {
        "category_col": cat_cols[0] if cat_cols else None,
        "second_cat_col": cat_cols[1] if len(cat_cols) > 1 else None,
        "value_col": numeric_cols[0] if numeric_cols else None,
        "target_col": numeric_cols[-1] if numeric_cols else None,
    }
    print("[自动检测] 分析列配置:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    return result
