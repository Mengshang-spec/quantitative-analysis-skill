# -*- coding: utf-8 -*-
"""
可视化模块：权重分布图（条形图+饼图）、权重矩阵热力图、相关热力图。
所有图表自动保存到 output/ 目录。
"""

import os
import matplotlib
matplotlib.use("Agg")  # 非交互后端，避免弹窗
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np


# ====== 中文字体设置 ======
def _setup_chinese_font():
    """尝试多个中文字体，找不到就用英文"""
    candidates = [
        "SimHei", "Microsoft YaHei", "WenQuanYi Zen Hei",
        "Noto Sans CJK SC", "PingFang SC", "STHeiti",
    ]
    available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return font
    print("[WARN] 未找到中文字体，图表中的中文可能显示为方块")
    plt.rcParams["axes.unicode_minus"] = False
    return None


_FONT = _setup_chinese_font()
sns.set_style("whitegrid")
sns.set_palette("viridis")


def _save_and_close(fig, save_path):
    """保存图表并关闭 figure"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[图表] 已保存: {save_path}")


def plot_weight_distribution(weight_df, category_col, weight_col="weight",
                             top_n=10, save_path=None):
    """
    绘制类别权重的条形图和饼图。

    参数:
        weight_df: category_frequency 返回的 DataFrame
        category_col: 类别列名
        weight_col: 权重列名
        top_n: 展示前 top_n 个类别
        save_path: 保存路径
    """
    data = weight_df.nlargest(top_n, weight_col).copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- 条形图 ---
    bars = sns.barplot(
        data=data, x=weight_col, y=category_col,
        hue=category_col, palette="viridis", legend=False, ax=ax1
    )
    ax1.set_title(f"{category_col} 权重分布（前 {top_n}）", fontsize=14, fontweight="bold")
    ax1.set_xlabel("权重")
    ax1.set_ylabel(category_col)

    # 柱子上标数值
    for bar, val in zip(bars.patches, data[weight_col]):
        ax1.text(bar.get_width() + bar.get_width() * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}", va="center", fontsize=9)

    # --- 饼图 ---
    wedges, texts, autotexts = ax2.pie(
        data[weight_col], labels=data[category_col],
        autopct="%1.1f%%", startangle=140,
        colors=sns.color_palette("viridis", len(data))
    )
    ax2.set_title(f"{category_col} 权重占比", fontsize=14, fontweight="bold")
    ax2.axis("equal")

    plt.tight_layout()
    if save_path:
        _save_and_close(fig, save_path)
    else:
        plt.show()


def plot_weight_matrix_heatmap(matrix, title="权重矩阵热力图",
                               fmt=".2f", cmap="Blues", save_path=None):
    """
    将权重矩阵绘制为热力图。

    参数:
        matrix: DataFrame 矩阵
        title: 标题
        fmt: 数值格式
        cmap: 色彩映射
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(max(10, matrix.shape[1] * 1.2),
                                    max(8, matrix.shape[0] * 0.8)))
    sns.heatmap(matrix, annot=True, fmt=fmt, cmap=cmap,
                linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        _save_and_close(fig, save_path)
    else:
        plt.show()


def plot_correlation_heatmap(corr_df, title="相关系数热力图", save_path=None):
    """
    画相关系数矩阵热力图。
    """
    fig, ax = plt.subplots(figsize=(max(12, corr_df.shape[0] * 0.8),
                                    max(10, corr_df.shape[0] * 0.7)))
    mask = np.triu(np.ones_like(corr_df, dtype=bool), k=1)
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, mask=mask, linewidths=0.5,
                ax=ax, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        _save_and_close(fig, save_path)
    else:
        plt.show()


def plot_prediction_results(y_true, y_pred, model_name="Model", save_path=None):
    """
    预测结果可视化：实际 vs 预测散点图 + 残差分布。
    """
    residuals = np.array(y_true) - np.array(y_pred)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 实际 vs 预测
    ax1.scatter(y_true, y_pred, alpha=0.5, edgecolors="k", linewidths=0.3)
    mn = min(min(y_true), min(y_pred))
    mx = max(max(y_true), max(y_pred))
    ax1.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="理想线")
    ax1.set_xlabel("实际值")
    ax1.set_ylabel("预测值")
    ax1.set_title(f"{model_name}: 实际 vs 预测", fontsize=13, fontweight="bold")
    ax1.legend()

    # 残差分布
    ax2.hist(residuals, bins=30, edgecolor="k", alpha=0.7, color="steelblue")
    ax2.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax2.set_xlabel("残差")
    ax2.set_ylabel("频数")
    ax2.set_title(f"{model_name}: 残差分布", fontsize=13, fontweight="bold")

    plt.tight_layout()
    if save_path:
        _save_and_close(fig, save_path)
    else:
        plt.show()
