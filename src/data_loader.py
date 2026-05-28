# -*- coding: utf-8 -*-
"""
数据加载模块：适配绝大部分 CSV 和 Excel（.xlsx / .xls）文件。
自动检测文件格式、编码，并提供统一的 DataFrame 入口。
"""

import os
import pandas as pd


def _detect_sep(filepath):
    """用常见分隔符试探 CSV，返回最可能的那个"""
    candidates = [",", "\t", ";", "|"]
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
    best, best_count = ",", 0
    for sep in candidates:
        cnt = first_line.count(sep)
        if cnt > best_count:
            best, best_count = sep, cnt
    return best


def load_csv(filepath, sep=None, encoding="utf-8"):
    """
    加载 CSV 文件。
    - sep: 分隔符，None 时自动检测
    - encoding: 先用给定编码，失败则尝试常见编码
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")

    encodings_to_try = [encoding, "utf-8", "gbk", "gb2312", "latin-1", "iso-8859-1"]
    if sep is None:
        sep = _detect_sep(filepath)

    for enc in encodings_to_try:
        try:
            df = pd.read_csv(filepath, sep=sep, encoding=enc)
            print(f"[OK] CSV 加载成功: {filepath}  (sep={sep!r}, encoding={enc})")
            return df
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue

    # 最后的 fallback
    df = pd.read_csv(filepath, sep=sep, encoding="utf-8", errors="replace")
    print(f"[WARN] CSV 使用 fallback 编码加载: {filepath}")
    return df


def load_excel(filepath, sheet_name=0):
    """
    加载 Excel 文件（.xlsx / .xls）。
    sheet_name: 0 = 第一张表，也可以写名字如 'Sheet1'
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".xls":
        engine = "xlrd"
    else:
        engine = "openpyxl"

    df = pd.read_excel(filepath, sheet_name=sheet_name, engine=engine)
    print(f"[OK] Excel 加载成功: {filepath}  (sheet={sheet_name})")
    return df


def load_data(filepath, sep=None, sheet_name=0, encoding="utf-8"):
    """
    统一入口：根据文件扩展名自动选择加载方式。
    支持 .csv / .tsv / .xlsx / .xls

    参数:
        filepath: 数据文件路径
        sep: CSV 分隔符（None = 自动检测）
        sheet_name: Excel 表名/索引
        encoding: 文件编码

    返回:
        pandas DataFrame
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext in (".xlsx", ".xls"):
        return load_excel(filepath, sheet_name=sheet_name)
    elif ext in (".csv", ".tsv", ".txt"):
        return load_csv(filepath, sep=sep, encoding=encoding)
    else:
        # 也当 CSV 尝试
        print(f"[WARN] 未知扩展名 {ext}，尝试以 CSV 格式加载")
        return load_csv(filepath, sep=sep, encoding=encoding)


def get_file_info(df):
    """打印 DataFrame 的基本信息，帮助用户了解数据"""
    print("=" * 50)
    print(f"数据形状: {df.shape[0]} 行 × {df.shape[1]} 列")
    print(f"列名: {list(df.columns)}")
    print(f"数据类型概览:\n{df.dtypes.value_counts().to_string()}")
    print(f"缺失值总数: {df.isnull().sum().sum()}")
    print("=" * 50)
    return df
