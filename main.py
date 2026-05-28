# -*- coding: utf-8 -*-
"""
定量数据分析主流程：
加载 → 清洗 → 基础分析 → 类别权重计算 → 矩阵构建 → 可视化 → 预测建模
适配绝大部分 CSV / XLSX 文件。
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from src.data_loader import load_data, get_file_info
from src.data_cleaner import clean_data
from src.analysis import (
    summary_stats, missing_report, correlation_matrix,
    column_type_report, auto_detect_analysis_columns
)
from src.quantitative_analysis import (
    category_frequency, build_weight_matrix,
    normalize_matrix, pair_wise_weight
)
from src.visualize import (
    plot_weight_distribution, plot_weight_matrix_heatmap,
    plot_correlation_heatmap, plot_prediction_results
)
from src.predict import run_prediction_pipeline


def main():
    print("=" * 60)
    print("     定量数据分析 Skill")
    print("=" * 60)

    # ====== 1. 加载数据 ======
    print(f"\n[1/6] 加载数据: {RAW_DATA_PATH}")
    if not os.path.exists(RAW_DATA_PATH):
        print(f"[ERROR] 数据文件不存在: {RAW_DATA_PATH}")
        print(f"请将 CSV/XLSX 文件放入: {RAW_DATA_DIR}")
        print("然后重新运行。程序将生成示例文件方便测试。")
        _create_sample_data()
        return

    df_raw = load_data(RAW_DATA_PATH, sep=CSV_SEP, sheet_name=XLSX_SHEET_NAME)
    get_file_info(df_raw)

    # ====== 2. 清洗数据 ======
    print("\n[2/6] 清洗数据")
    df = clean_data(
        df_raw,
        drop_dup=DROP_DUPLICATES,
        drop_threshold=DROP_NA_THRESHOLD,
        num_strategy=FILL_NUMERIC_STRATEGY,
        cat_strategy=FILL_CATEGORY_STRATEGY,
    )
    df.to_csv(CLEANED_DATA_PATH, index=False, encoding="utf-8-sig")
    print(f"清洗后数据已保存: {CLEANED_DATA_PATH}")

    # 自动检测分析列
    auto_cols = auto_detect_analysis_columns(df)

    # ====== 3. 基础分析 ======
    print("\n[3/6] 基础描述性分析")
    summary_stats(df)
    missing_report(df)

    corr = correlation_matrix(df, method="pearson")
    if not corr.empty:
        corr_path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
        plot_correlation_heatmap(corr, save_path=corr_path)

    # ====== 4. 定量分析：类别权重 ======
    print("\n[4/6] 定量分析：类别权重计算")

    # 使用配置中的列，如果为 None 则用自动检测的
    cat_col = CATEGORY_COL or auto_cols["category_col"]
    second_cat = MATRIX_COL_COL or auto_cols["second_cat_col"]
    value_col = MATRIX_VALUE_COL or auto_cols["value_col"]

    if cat_col is None:
        print("[SKIP] 未找到分类列，跳过类别权重分析。")
    else:
        print(f"使用分类列: {cat_col}")
        weight_df = category_frequency(df, cat_col, WEIGHT_METHOD)

        # 画权重分布图
        weight_path = os.path.join(OUTPUT_DIR, "weight_distribution.png")
        plot_weight_distribution(
            weight_df, cat_col, weight_col="weight",
            top_n=TOP_N_CATEGORIES, save_path=weight_path
        )

        # 如果有数值列，做一次成对权重
        if value_col:
            print(f"\n成对权重: 按 {cat_col} 分组聚合 {value_col}")
            pair_wise_weight(df, cat_col, value_col, agg_func="sum")

        # ====== 5. 权重矩阵 ======
        print("\n[5/6] 构建权重矩阵")
        if second_cat:
            print(f"矩阵维度: {cat_col} × {second_cat}")
            matrix = build_weight_matrix(df, cat_col, second_cat,
                                         value_col=value_col, agg_func=MATRIX_AGG_FUNC)

            norm_matrix = normalize_matrix(matrix, method="total")

            matrix_path = os.path.join(OUTPUT_DIR, "weight_matrix_heatmap.png")
            plot_weight_matrix_heatmap(
                norm_matrix, title=f"归一化权重矩阵: {cat_col} × {second_cat}",
                save_path=matrix_path
            )
        else:
            print("[SKIP] 只有一个分类列，跳过二维矩阵构建。如需矩阵请设置 MATRIX_COL_COL。")

        # ====== 6. 预测建模 ======
    print("\n[6/6] 预测建模")
    if not DO_PREDICTION:
        print("[SKIP] DO_PREDICTION = False，跳过预测步骤")
    else:
        target_col = TARGET_COL or auto_cols["target_col"]
        if target_col is None:
            print("[SKIP] 未找到目标列，跳过预测。请设置 TARGET_COL。")
        else:
            print(f"目标列: {target_col}, 模型: {MODEL_TYPE}")
            try:
                pred_result = run_prediction_pipeline(
                    df, target_col, model_type=MODEL_TYPE,
                    test_size=TEST_SIZE, random_seed=RANDOM_SEED,
                    save_dir=OUTPUT_DIR
                )

                # 全模型对比结果导出 CSV
                if pred_result.get("comparison_df") is not None:
                    cmp_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
                    pred_result["comparison_df"].to_csv(cmp_path, index=False, encoding="utf-8-sig")
                    print(f"[导出] 全模型对比: {cmp_path}")

                # 预测结果可视化（仅回归有散点图）
                if not pred_result["is_classification"]:
                    pred_path = os.path.join(OUTPUT_DIR, "prediction_results.png")
                    plot_prediction_results(
                        pred_result["y_test"], pred_result["y_pred"],
                        model_name=MODEL_TYPE, save_path=pred_path
                    )

            except Exception as e:
                print(f"[ERROR] 预测建模失败: {e}")
                import traceback; traceback.print_exc()
    # ====== 完成 ======
    print("\n" + "=" * 60)
    print("     定量分析流程全部完成！")
    print(f"     输出目录: {OUTPUT_DIR}")
    _list_output_files()
    print("=" * 60)


def _list_output_files():
    """列出输出目录中的文件"""
    files = os.listdir(OUTPUT_DIR)
    if files:
        print("生成的文件:")
        for f in sorted(files):
            fpath = os.path.join(OUTPUT_DIR, f)
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  - {f} ({size_kb:.1f} KB)")


def _create_sample_data():
    """当数据文件不存在时，生成一份示例数据方便测试。"""
    import pandas as pd
    import numpy as np

    np.random.seed(42)
    n = 500

    categories = ["电子产品", "服装", "食品", "家居", "运动户外", "图书"]
    regions = ["华东", "华南", "华北", "西南", "华中"]

    sample = pd.DataFrame({
        "category": np.random.choice(categories, n),
        "region": np.random.choice(regions, n),
        "sales": np.random.randint(100, 10000, n),
        "quantity": np.random.randint(1, 50, n),
        "rating": np.round(np.random.uniform(1, 5, n), 1),
        "profit": np.round(np.random.uniform(-500, 5000, n), 2),
        "customer_age": np.random.randint(18, 70, n),
    })

    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    sample_path = os.path.join(RAW_DATA_DIR, "sample_data.csv")
    sample.to_csv(sample_path, index=False, encoding="utf-8-sig")
    print(f"\n[INFO] 已生成示例数据: {sample_path}")
    print("请重新运行 main.py 进行分析。")


if __name__ == "__main__":
    main()

