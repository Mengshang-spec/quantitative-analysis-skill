# -*- coding: utf-8 -*-
"""
全局配置：路径、分析参数集中管理。
支持自动检测数据文件，适配绝大部分 CSV / XLSX。
"""

import os
import glob

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ====== 数据路径 —— 自动扫描 data/raw/ 下第一个 csv/xlsx/xls 文件 ======
RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
CLEANED_DATA_PATH = os.path.join(ROOT_DIR, "data", "cleaned_data.csv")

def _auto_detect_data_path():
    """自动在 data/raw/ 下找到第一个支持的表格文件"""
    candidates = []
    for ext in ["*.csv", "*.xlsx", "*.xls"]:
        candidates.extend(glob.glob(os.path.join(RAW_DATA_DIR, ext)))
    candidates.sort()
    return candidates[0] if candidates else os.path.join(RAW_DATA_DIR, "survey_data.csv")

RAW_DATA_PATH = _auto_detect_data_path()

# ====== 数据格式参数 ======
CSV_SEP = ","              # CSV 分隔符，常见的有 ',' '\t' ';'
CSV_ENCODING = "utf-8"     # CSV 文件编码
XLSX_SHEET_NAME = 0        # Excel 的 sheet 名：0=第一张表，也可写具体名字
JSON_LINES = False
DB_CONNECTION_STRING = "sqlite:///data/raw/db.sqlite"
SQL_QUERY = "SELECT * FROM main_table"

# ====== 清洗参数 ======
DROP_DUPLICATES = True          # 是否去重
FILL_NUMERIC_STRATEGY = "median"  # 数值缺失值填充策略： "median" / "mean" / "zero" / None
FILL_CATEGORY_STRATEGY = "mode"   # 分类缺失值填充策略： "mode" / "unknown" / None
DROP_NA_THRESHOLD = 0.5           # 若某列缺失比例超过此值，直接删除该列

# ====== 定量分析参数 ======
# 如果设为 None，程序会自动选择第一列 object/category 列作为分类列
CATEGORY_COL = None
WEIGHT_METHOD = "frequency"      # "frequency"(频数) / "proportion"(占比)
MATRIX_ROW_COL = None            # 矩阵行分类列，None=自动选
MATRIX_COL_COL = None            # 矩阵列分类列，None=自动选
MATRIX_VALUE_COL = None          # 聚合数值列，None=用计数
MATRIX_AGG_FUNC = "count"        # 矩阵聚合函数：count / sum / mean / median
TOP_N_CATEGORIES = 10            # 绘图中展示前 N 个类别

# ====== 预测建模参数 ======
TARGET_COL = None                # 预测目标列，None=自动选最后一列数值列
MODEL_TYPE = "all"                # "all"全模型对比 / "random_forest" / "linear" / "xgboost" / "decision_tree" / "svm" / "knn" / "mlp" / ...
RANDOM_SEED = 42
TEST_SIZE = 0.2
DO_PREDICTION = True             # 是否执行预测建模步骤

# ====== 输出目录 ======
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

