# -*- coding: utf-8 -*-
"""
定量分析核心模块：
按类别统计 → 计算权重（频数/占比） → 构建权重矩阵 → 归一化。
所有函数返回计算结果，便于后续可视化和进一步建模。
"""

import pandas as pd
import numpy as np


def category_frequency(df, category_col, weight_method="frequency"):
    """
    按指定列进行类别分析，计算各类别的权重。

    参数:
        df: DataFrame
        category_col: 分类列名，例如 "产品类型"
        weight_method: "frequency" 返回频数， "proportion" 返回占比

    返回:
        DataFrame，列: category, count, weight
    """
    if category_col not in df.columns:
        raise KeyError(f"列 '{category_col}' 不在数据中，可用列: {list(df.columns)}")

    freq_df = df[category_col].value_counts().reset_index()
    freq_df.columns = [category_col, "count"]

    if weight_method == "proportion":
        total = freq_df["count"].sum()
        freq_df["weight"] = freq_df["count"] / total
        print(f"\n类别 '{category_col}' 的占比权重：")
    else:
        freq_df["weight"] = freq_df["count"]
        print(f"\n类别 '{category_col}' 的频数权重：")

    print(freq_df.to_string(index=False))

    # 找出权重最高的类别
    top_idx = freq_df["weight"].idxmax()
    top = freq_df.loc[top_idx]
    print(f">>> 权重最大的类别: [{top[category_col]}], 权重 = {top['weight']}")

    return freq_df


def build_weight_matrix(df, row_col, col_col, value_col=None, agg_func="count"):
    """
    构建二维权重矩阵（交叉表/透视表）。

    参数:
        df: DataFrame
        row_col: 矩阵行索引
        col_col: 矩阵列索引
        value_col: 聚合数值列，None = 计频数
        agg_func: "count" / "sum" / "mean" / "median"

    返回:
        DataFrame 矩阵
    """
    for c in [row_col, col_col]:
        if c not in df.columns:
            raise KeyError(f"列 '{c}' 不在数据中")

    if value_col is None:
        matrix = pd.crosstab(df[row_col], df[col_col])
        print(f"\n构建计数矩阵：{row_col} × {col_col}")
    else:
        if value_col not in df.columns:
            raise KeyError(f"数值列 '{value_col}' 不存在")
        matrix = df.pivot_table(
            index=row_col, columns=col_col,
            values=value_col, aggfunc=agg_func, fill_value=0
        )
        print(f"\n构建聚合矩阵：{row_col} × {col_col}，{agg_func}({value_col})")

    print(matrix.round(2).to_string())
    return matrix


def normalize_matrix(matrix, method="total"):
    """
    归一化矩阵。

    参数:
        matrix: 原始权重矩阵
        method: "total" 所有元素和为 1，"row" 每行和为 1

    返回:
        归一化后的矩阵
    """
    if method == "total":
        normed = matrix / matrix.sum().sum()
        print("\n矩阵按总和归一化（所有元素之和=1）")
    elif method == "row":
        row_sums = matrix.sum(axis=1)
        row_sums = row_sums.replace(0, 1)  # 避免除以 0
        normed = matrix.div(row_sums, axis=0)
        print("\n矩阵按行归一化（每行之和=1）")
    else:
        raise ValueError("method 只能是 'total' 或 'row'")

    print(normed.round(4).to_string())
    return normed


def pair_wise_weight(df, category_col, value_col, agg_func="sum"):
    """
    成对比较权重：计算每个类别在某个数值指标上的占比，类似 AHP 思路。

    返回权重 Series，按从大到小排序。
    """
    grouped = df.groupby(category_col)[value_col].agg(agg_func)
    total = grouped.sum()
    weights = (grouped / total).sort_values(ascending=False)
    print(f"\n按 {agg_func}({value_col}) 计算的成对权重：")
    for cat, w in weights.items():
        print(f"  {cat}: {w:.4f} ({w*100:.2f}%)")
    return weights
