# -*- coding: utf-8 -*-
"""
预测建模模块：
自动编码分类变量 → 划分训练/测试集 → 训练模型 → 评估 → 特征重要性。
支持 Random Forest、Linear Regression、XGBoost。
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def _is_classification(y):
    """判断是分类还是回归：类别数 <= 15 且非浮点则认为分类"""
    if y.dtype == "object" or y.dtype.name == "category":
        return True
    unique_ratio = y.nunique() / len(y)
    return unique_ratio < 0.05 and y.nunique() <= 15


def _encode_categorical(df):
    """
    将分类列转换为数值。返回编码后的 DataFrame 和编码器字典。
    """
    df_encoded = df.copy()
    encoders = {}
    for col in df_encoded.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le
    return df_encoded, encoders


def prepare_data(df, target_col):
    """
    准备训练数据：分离特征和目标、编码分类变量、划分训练/测试集。

    返回:
        X_train, X_test, y_train, y_test, feature_names, is_classification
    """
    if target_col not in df.columns:
        raise KeyError(f"目标列 '{target_col}' 不在数据中，可用列: {list(df.columns)}")

    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()

    # 编码分类特征
    X_encoded, encoders = _encode_categorical(X)

    # 目标列如果是 object/category 也要编码
    if y.dtype == "object" or y.dtype.name == "category":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)), name=target_col)

    # 判断分类 / 回归
    classification = _is_classification(y)
    task = "分类" if classification else "回归"
    print(f"\n[预测任务] 目标列: {target_col}, 任务类型: {task}, 样本数: {len(y)}")

    feature_names = X_encoded.columns.tolist()

    # 划分
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )
    print(f"训练集: {len(X_train)} 条, 测试集: {len(X_test)} 条")

    return X_train, X_test, y_train, y_test, feature_names, classification


def train_model(X_train, y_train, model_type="random_forest", classification=False):
    """
    训练模型。

    参数:
        model_type: "random_forest" / "linear" / "xgboost"
        classification: True = 分类，False = 回归

    返回:
        训练好的模型
    """
    if model_type == "random_forest":
        if classification:
            model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )
        else:
            model = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )
        print(f"[模型] Random Forest ({'分类' if classification else '回归'})")

    elif model_type == "linear":
        if classification:
            model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            model = LinearRegression()
        print(f"[模型] Linear Model ({'分类' if classification else '回归'})")

    elif model_type == "xgboost":
        try:
            from xgboost import XGBRegressor, XGBClassifier
        except ImportError:
            print("[ERROR] xgboost 未安装，回退到 Random Forest。请 `pip install xgboost`")
            return train_model(X_train, y_train, "random_forest", classification)

        if classification:
            model = XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=42, use_label_encoder=False, eval_metric="logloss"
            )
        else:
            model = XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
            )
        print(f"[模型] XGBoost ({'分类' if classification else '回归'})")

    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    model.fit(X_train, y_train)
    print("[模型] 训练完成")
    return model


def evaluate_model(model, X_test, y_test, classification=False):
    """
    评估模型并打印指标。

    返回:
        dict: 各指标值
    """
    y_pred = model.predict(X_test)
    results = {}

    print("\n===== 模型评估 =====")
    if classification:
        results["accuracy"] = accuracy_score(y_test, y_pred)
        results["precision"] = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        results["recall"] = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        results["f1"] = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        for k, v in results.items():
            print(f"  {k}: {v:.4f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred, zero_division=0))
    else:
        results["mse"] = mean_squared_error(y_test, y_pred)
        results["rmse"] = np.sqrt(results["mse"])
        results["mae"] = mean_absolute_error(y_test, y_pred)
        results["r2"] = r2_score(y_test, y_pred)

        for k, v in results.items():
            print(f"  {k}: {v:.4f}")

    return results, y_pred


def feature_importance(model, feature_names, top_n=20, save_path=None):
    """
    输出并绘制特征重要性（仅对树模型有效）。
    """
    if not hasattr(model, "feature_importances_"):
        print("[INFO] 该模型不支持特征重要性")
        return None

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    imp_df = pd.DataFrame({
        "feature": [feature_names[i] for i in indices],
        "importance": importances[indices]
    })

    print(f"\n===== 特征重要性（前 {top_n}）=====")
    for _, row in imp_df.iterrows():
        print(f"  {row['feature']:30s} {row['importance']:.4f}")

    # 绘图
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
        sns.barplot(data=imp_df, x="importance", y="feature", hue="feature", palette="viridis", legend=False, ax=ax)
        ax.set_title("特征重要性", fontsize=14, fontweight="bold")
        ax.set_xlabel("重要性")
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"[图表] 特征重要性图已保存: {save_path}")
        else:
            plt.show()
    except Exception as e:
        print(f"[WARN] 特征重要性图绘制失败: {e}")

    return imp_df


def run_prediction_pipeline(df, target_col, model_type="random_forest",
                            test_size=0.2, random_seed=42, save_dir=None):
    """
    一站式预测流水线：准备数据 → 训练 → 评估 → 特征重要性 → 可视化。

    返回:
        dict: 包含模型、评估结果、预测值等
    """
    X_train, X_test, y_train, y_test, feature_names, is_clf = prepare_data(df, target_col)

    model = train_model(X_train, y_train, model_type, classification=is_clf)
    results, y_pred = evaluate_model(model, X_test, y_test, classification=is_clf)

    # 特征重要性
    save_path = os.path.join(save_dir, "feature_importance.png") if save_dir else None
    imp_df = feature_importance(model, feature_names, save_path=save_path)

    return {
        "model": model,
        "results": results,
        "y_test": y_test,
        "y_pred": y_pred,
        "feature_names": feature_names,
        "is_classification": is_clf,
        "feature_importance": imp_df,
    }

