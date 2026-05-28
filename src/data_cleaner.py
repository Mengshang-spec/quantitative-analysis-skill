# -*- coding: utf-8 -*-
"""
数据清洗模块：处理缺失值、重复值、异常值，让数据适合定量分析。
"""

import pandas as pd
import numpy as np


def drop_high_missing(df, threshold=0.5):
    """
    删除缺失比例超过 threshold 的列。
    返回清理后的 DataFrame 和被删列的列表。
    """
    missing_ratio = df.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
    if cols_to_drop:
        print(f"[清洗] 删除缺失率 > {threshold:.0%} 的列: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    return df, cols_to_drop


def drop_constant_columns(df):
    """删除只有单一值（无信息量）的列"""
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if constant_cols:
        print(f"[清洗] 删除常数列（只有单一值）: {constant_cols}")
        df = df.drop(columns=constant_cols)
    return df


def fill_missing(df, num_strategy="median", cat_strategy="mode"):
    """
    填充缺失值。

    num_strategy: "median" / "mean" / "zero" / None（不填）
    cat_strategy: "mode" / "unknown" / None（不填）
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    # 数值列填充
    if num_strategy and len(numeric_cols) > 0:
        for col in numeric_cols:
            if df[col].isnull().sum() == 0:
                continue
            if num_strategy == "median":
                fill_val = df[col].median()
            elif num_strategy == "mean":
                fill_val = df[col].mean()
            elif num_strategy == "zero":
                fill_val = 0
            else:
                continue
            df[col] = df[col].fillna(fill_val)
        print(f"[清洗] 数值列用 {num_strategy} 填充完成，涉及 {len(numeric_cols)} 列")

    # 分类列填充
    if cat_strategy and len(cat_cols) > 0:
        for col in cat_cols:
            if df[col].isnull().sum() == 0:
                continue
            if cat_strategy == "mode":
                mode_val = df[col].mode()
                fill_val = mode_val[0] if len(mode_val) > 0 else "missing"
            elif cat_strategy == "unknown":
                fill_val = "unknown"
            else:
                continue
            df[col] = df[col].fillna(fill_val)
        print(f"[清洗] 分类列用 {cat_strategy} 填充完成，涉及 {len(cat_cols)} 列")

    return df


def remove_outliers_iqr(df, multiplier=3.0):
    """
    用 IQR 法标记（不删除）极端异常值，设为 NaN 后再用 fill_missing 处理。
    只作用于数值列。
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_count = 0
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            continue
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        mask = (df[col] < lower) | (df[col] > upper)
        outlier_count += mask.sum()
        df.loc[mask, col] = np.nan
    if outlier_count > 0:
        print(f"[清洗] 用 IQR 法标记了 {outlier_count} 个异常值为 NaN")
    return df


def clean_data(df,
               drop_dup=True,
               drop_threshold=0.5,
               num_strategy="median",
               cat_strategy="mode",
               remove_outliers=False):
    """
    一站式数据清洗流水线。

    参数:
        df: 原始 DataFrame
        drop_dup: 是否去重
        drop_threshold: 缺失率阈值，高于此值的列直接删除
        num_strategy: 数值缺失填充策略
        cat_strategy: 分类缺失填充策略
        remove_outliers: 是否处理异常值

    返回清洗后的 DataFrame
    """
    print("\n===== 开始数据清洗 =====")
    print(f"清洗前形状: {df.shape}")

    # 1. 去重
    if drop_dup:
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        if before != after:
            print(f"[清洗] 删除了 {before - after} 条重复行")

    # 2. 删缺失率过高的列
    df, dropped = drop_high_missing(df, drop_threshold)

    # 3. 删常数列
    df = drop_constant_columns(df)

    # 4. 异常值处理（可选）
    if remove_outliers:
        df = remove_outliers_iqr(df)

    # 5. 填充缺失值
    df = fill_missing(df, num_strategy=num_strategy, cat_strategy=cat_strategy)

    # 6. 若还有少量缺失值，兜底删除含 NaN 的行
    remaining = df.isnull().sum().sum()
    if remaining > 0:
        before = len(df)
        df = df.dropna()
        after = len(df)
        print(f"[清洗] 兜底删除了 {before - after} 条仍含缺失的行")

    print(f"清洗后形状: {df.shape}")
    print("===== 数据清洗完成 =====\n")
    return df
