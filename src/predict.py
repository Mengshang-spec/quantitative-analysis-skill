# -*- coding: utf-8 -*-
"""
预测建模模块：全覆盖有监督学习算法。
分类/回归自动判别 → 编码 → 标准化 → 训练 → 评估 → 模型对比。

支持的模型:
    回归: Linear, Ridge, Lasso, DecisionTree, RandomForest, ExtraTrees,
          GradientBoosting, AdaBoost, SVR, KNN, MLP, XGBoost, LightGBM
    分类: Logistic, DecisionTree, RandomForest, ExtraTrees,
          GradientBoosting, AdaBoost, SVC, KNN, GaussianNB, MLP,
          XGBoost, LightGBM
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ======================================================================
#  工具函数
# ======================================================================

def _is_classification(y):
    """判断分类/回归: object/category 类型 或 类别 ≤15 且占比 <5%"""
    if y.dtype == "object" or y.dtype.name == "category":
        return True
    unique_ratio = y.nunique() / len(y)
    return unique_ratio < 0.05 and y.nunique() <= 15


def _encode_categorical(df):
    """LabelEncode 所有分类列，返回 (编码后df, 编码器dict)"""
    df_out = df.copy()
    encoders = {}
    for col in df_out.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        df_out[col] = le.fit_transform(df_out[col].astype(str))
        encoders[col] = le
    return df_out, encoders


def prepare_data(df, target_col):
    """分离特征/目标 → 编码 → 划分 → 返回 (X_train, X_test, y_train, y_test, feature_names, is_clf)"""
    if target_col not in df.columns:
        raise KeyError(f"目标列 '{target_col}' 不在数据中，可用列: {list(df.columns)}")

    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()

    X_encoded, _ = _encode_categorical(X)

    if y.dtype == "object" or y.dtype.name == "category":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)), name=target_col)

    is_clf = _is_classification(y)
    task = "分类" if is_clf else "回归"
    print(f"\n[预测任务] 目标列: {target_col}  任务: {task}  样本: {len(y)}")

    feature_names = X_encoded.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )
    print(f"训练集: {len(X_train)}  测试集: {len(X_test)}")

    return X_train, X_test, y_train, y_test, feature_names, is_clf


# ======================================================================
#  模型工厂 —— 所有有监督学习算法
# ======================================================================

def _get_model(model_type, classification):
    """
    返回 (model, needs_scaling) 元组。
    needs_scaling=True 表示训练前需要 StandardScaler（SVM/KNN/MLP/Linear/Lasso/Ridge/Logistic）
    """
    clf = classification
    mt = model_type.lower()

    # ---- 线性模型 ----
    if mt == "linear":
        if clf:  return LogisticRegression(max_iter=2000, random_state=42), True
        else:    return LinearRegression(), True
    if mt == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1.0, random_state=42), True
    if mt == "lasso":
        from sklearn.linear_model import Lasso
        return Lasso(alpha=0.01, random_state=42, max_iter=5000), True

    # ---- 树模型 ----
    if mt == "decision_tree":
        from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
        if clf:  return DecisionTreeClassifier(max_depth=10, random_state=42), False
        else:    return DecisionTreeRegressor(max_depth=10, random_state=42), False
    if mt == "random_forest":
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        if clf:  return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1), False
        else:    return RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1), False
    if mt == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
        if clf:  return ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1), False
        else:    return ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1), False

    # ---- 提升模型 ----
    if mt == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
        if clf:  return GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42), False
        else:    return GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42), False
    if mt == "adaboost":
        from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
        if clf:  return AdaBoostClassifier(n_estimators=100, random_state=42, algorithm="SAMME"), False
        else:    return AdaBoostRegressor(n_estimators=100, random_state=42), False
    if mt == "xgboost":
        try:
            from xgboost import XGBRegressor, XGBClassifier
        except ImportError:
            print("[WARN] xgboost 未安装，回退 RandomForest")
            return _get_model("random_forest", classification)
        if clf:  return XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric="logloss"), False
        else:    return XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42), False
    if mt == "lightgbm":
        try:
            from lightgbm import LGBMRegressor, LGBMClassifier
        except ImportError:
            print("[WARN] lightgbm 未安装，回退 RandomForest")
            return _get_model("random_forest", classification)
        if clf:  return LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1), False
        else:    return LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1), False

    # ---- SVM ----
    if mt == "svm":
        from sklearn.svm import SVR, SVC
        if clf:  return SVC(kernel="rbf", C=1.0, random_state=42, probability=True), True
        else:    return SVR(kernel="rbf", C=1.0), True

    # ---- KNN ----
    if mt == "knn":
        from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
        if clf:  return KNeighborsClassifier(n_neighbors=5, n_jobs=-1), True
        else:    return KNeighborsRegressor(n_neighbors=5, n_jobs=-1), True

    # ---- 朴素贝叶斯（仅分类）----
    if mt == "naive_bayes":
        if not clf:
            print("[WARN] NaiveBayes 仅支持分类，已回退 RandomForest")
            return _get_model("random_forest", classification)
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB(), True

    # ---- 神经网络 ----
    if mt == "mlp":
        from sklearn.neural_network import MLPRegressor, MLPClassifier
        if clf:  return MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, early_stopping=True), True
        else:    return MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, early_stopping=True), True

    raise ValueError(f"不支持的模型: '{model_type}'。可用: linear, ridge, lasso, decision_tree, random_forest, extra_trees, gradient_boosting, adaboost, xgboost, lightgbm, svm, knn, naive_bayes, mlp, all")


# ======================================================================
#  训练 & 评估
# ======================================================================

def train_model(X_train, y_train, model_type="random_forest", classification=False):
    """单模型训练，自动处理标准化"""
    model, needs_scaling = _get_model(model_type, classification)

    if needs_scaling:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

    model.fit(X_train, y_train)
    print(f"[模型] {model_type} 训练完成{' (已标准化)' if needs_scaling else ''}")
    return model, needs_scaling


def evaluate_model(model, X_test, y_test, classification=False, needs_scaling=False):
    """评估并返回 (results_dict, y_pred)"""
    if needs_scaling:
        from sklearn.preprocessing import StandardScaler
        X_test = StandardScaler().fit_transform(X_test)

    y_pred = model.predict(X_test)
    results = {}

    print("\n----- 模型评估 -----")
    if classification:
        results["accuracy"]  = accuracy_score(y_test, y_pred)
        results["precision"] = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        results["recall"]    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        results["f1"]        = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        for k, v in results.items():
            print(f"  {k:12s}: {v:.4f}")
        print("\n" + classification_report(y_test, y_pred, zero_division=0))
    else:
        results["mse"]  = mean_squared_error(y_test, y_pred)
        results["rmse"] = np.sqrt(results["mse"])
        results["mae"]  = mean_absolute_error(y_test, y_pred)
        results["r2"]   = r2_score(y_test, y_pred)
        for k, v in results.items():
            print(f"  {k:6s}: {v:.4f}")

    return results, y_pred


# ======================================================================
#  特征重要性
# ======================================================================

def feature_importance(model, feature_names, top_n=20, save_path=None):
    """输出并绘制特征重要性（树模型/线性模型可用）"""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        importances = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)
    else:
        if hasattr(model, "named_steps"):
            for step_name in ["regressor", "classifier"]:
                if hasattr(model.named_steps.get(step_name, None), "feature_importances_"):
                    importances = model.named_steps[step_name].feature_importances_
                    break
            else:
                print("[INFO] 该模型不支持特征重要性")
                return None
        else:
            print("[INFO] 该模型不支持特征重要性")
            return None

    indices = np.argsort(importances)[::-1][:top_n]
    imp_df = pd.DataFrame({
        "feature": [feature_names[i] for i in indices],
        "importance": importances[indices]
    })

    print(f"\n----- 特征重要性（前 {top_n}）-----")
    for _, row in imp_df.iterrows():
        print(f"  {row['feature']:30s} {row['importance']:.4f}")

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
            print(f"[图表] 已保存: {save_path}")
    except Exception as e:
        print(f"[WARN] 特征重要性图失败: {e}")

    return imp_df


# ======================================================================
#  全模型对比 —— 一次训练所有算法，选出最优
# ======================================================================

def _cv_score(model, X, y, classification, cv=5):
    """Cross-validation 评分（回归用 R2, 分类用 accuracy）"""
    try:
        scoring = "accuracy" if classification else "r2"
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        return scores.mean()
    except Exception:
        return np.nan


def run_all_models(X_train, y_train, X_test, y_test, classification, save_dir=None):
    """
    训练所有可用算法，对比评估指标，绘制对比图，返回排名 DataFrame。
    """
    models_to_try = [
        "random_forest", "extra_trees", "gradient_boosting", "decision_tree",
        "adaboost", "linear", "ridge", "lasso", "svm", "knn", "mlp",
    ]
    if classification:
        models_to_try.append("naive_bayes")

    # 可选模型
    for m in ["xgboost", "lightgbm"]:
        try:
            _get_model(m, classification)
            models_to_try.append(m)
        except (ImportError, ValueError):
            pass

    print(f"\n{'='*60}")
    print(f"  全模型对比（{len(models_to_try)} 个算法）")
    print(f"{'='*60}")

    rows = []
    for mt in models_to_try:
        try:
            model, needs_scaling = _get_model(mt, classification)
            X_tr = StandardScaler().fit_transform(X_train) if needs_scaling else X_train
            X_te = StandardScaler().fit_transform(X_test)  if needs_scaling else X_test

            model.fit(X_tr, y_train)
            y_pred = model.predict(X_te)

            cv_mean = _cv_score(model, X_tr, y_train, classification)

            if classification:
                rows.append({
                    "model": mt,
                    "accuracy":  accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                    "recall":    recall_score(y_test, y_pred, average="weighted", zero_division=0),
                    "f1":        f1_score(y_test, y_pred, average="weighted", zero_division=0),
                    "cv_score":  cv_mean,
                })
            else:
                rows.append({
                    "model": mt,
                    "r2":   r2_score(y_test, y_pred),
                    "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                    "mae":  mean_absolute_error(y_test, y_pred),
                    "cv_score": cv_mean,
                })
            print(f"  {mt:20s}  done")
        except Exception as e:
            print(f"  {mt:20s}  FAILED: {e}")

    df = pd.DataFrame(rows)

    # 排序: 分类按 f1，回归按 r2
    sort_col = "f1" if classification else "r2"
    df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)

    print(f"\n----- 模型排名（按 {sort_col}）-----")
    print(df.round(4).to_string(index=False))

    # 绘图
    _plot_model_comparison(df, classification, save_dir)

    return df


def _plot_model_comparison(comp_df, classification, save_dir=None):
    """模型对比柱状图"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        metric = "f1" if classification else "r2"
        title_metric = "F1 Score" if classification else "R²"
        data = comp_df.sort_values(metric)

        fig, ax = plt.subplots(figsize=(10, max(6, len(data) * 0.4)))
        bars = sns.barplot(data=data, x=metric, y="model", hue="model",
                           palette="viridis", legend=False, ax=ax)
        ax.set_title(f"模型对比 —— {title_metric}", fontsize=14, fontweight="bold")
        ax.set_xlabel(title_metric)
        ax.set_ylabel("")

        for bar, val in zip(bars.patches, data[metric]):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9)

        plt.tight_layout()
        if save_dir:
            path = os.path.join(save_dir, "model_comparison.png")
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"[图表] 模型对比图已保存: {path}")
    except Exception as e:
        print(f"[WARN] 模型对比图失败: {e}")


# ======================================================================
#  对外入口
# ======================================================================

def run_prediction_pipeline(df, target_col, model_type="random_forest",
                            test_size=0.2, random_seed=42, save_dir=None):
    """
    一站式预测流水线。

    model_type="all" 时运行全模型对比。

    返回 dict: {model, results, y_test, y_pred, feature_names, is_classification, feature_importance, comparison_df}
    """
    X_train, X_test, y_train, y_test, feature_names, is_clf = prepare_data(df, target_col)

    # ---- 全模型对比模式 ----
    if model_type == "all":
        comp_df = run_all_models(X_train, y_train, X_test, y_test, is_clf, save_dir)
        # 用最优模型做单模型评估
        best_model_name = comp_df.iloc[0]["model"]
        print(f"\n>>> 最优模型: {best_model_name}，使用该模型做详细评估")
        model, needs_scaling = train_model(X_train, y_train, best_model_name, is_clf)
        results, y_pred = evaluate_model(model, X_test, y_test, is_clf, needs_scaling)
    else:
        model, needs_scaling = train_model(X_train, y_train, model_type, is_clf)
        results, y_pred = evaluate_model(model, X_test, y_test, is_clf, needs_scaling)
        comp_df = None

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
        "comparison_df": comp_df,
    }

