# 定量数据分析 Skill

以**定量分析**为核心的 Python 数据分析工具包：按类别统计 → 计算权重 → 构建矩阵 → 绘制权重分布图 → 全模型算法预测。

适配绝大部分 CSV / XLSX 文件，自动检测编码、分隔符、列类型，零配置即可运行。

---

## 核心流程

```
加载数据 → 清洗 → 基础统计 → 类别权重 → 权重矩阵 → 可视化 → 全模型预测对比
```

1. **数据加载** — 自动识别 .csv / .xlsx / .xls，探测编码和分隔符
2. **数据清洗** — 去重、缺失值填充（均值/中位数/众数）、异常值处理
3. **基础描述性统计** — 摘要统计、缺失值报告、相关系数矩阵
4. **类别权重计算** — 按类别统计频数/占比，找出权重最大的类别
5. **权重矩阵构建** — 二维交叉矩阵 + 归一化
6. **权重分布图** — 条形图 + 饼图 + 矩阵热力图
7. **全模型预测对比** — 一次训练 13 种算法，自动排名选最优，输出对比图

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 放入数据

把你的 CSV 或 Excel 文件放入 `data/raw/` 目录。

### 3. 运行

```bash
python main.py
```

如果没有数据文件，程序会自动生成一份 `sample_data.csv` 供你测试。

---

## 文件结构

```
quantitative_analysis_skill/
├── README.md                          # 本文件
├── requirements.txt                   # Python 依赖
├── config.py                          # 全局配置（路径、分析参数）
├── main.py                            # 主入口，串联全流程
├── data/
│   └── raw/                           # 放置原始数据文件
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # 多格式加载（CSV/Excel 自动适配）
│   ├── data_cleaner.py                # 清洗（缺失值、重复值、异常值）
│   ├── analysis.py                    # 基础描述性分析 + 列类型检测
│   ├── quantitative_analysis.py       # 定量分析核心：权重计算、矩阵构建
│   ├── visualize.py                   # 可视化：权重分布图、热力图、预测图
│   └── predict.py                     # 预测建模：13 种有监督学习算法
├── output/                            # 所有图表和结果输出
│   ├── weight_distribution.png        # 类别权重分布图（条形+饼图）
│   ├── weight_matrix_heatmap.png      # 权重矩阵热力图
│   ├── correlation_heatmap.png        # 相关系数热力图
│   ├── model_comparison.png           # 全模型对比排名图
│   ├── model_comparison.csv           # 全模型对比数据表
│   ├── prediction_results.png         # 预测结果：实际 vs 预测 + 残差
│   └── feature_importance.png         # 特征重要性排名
└── notebooks/
    └── exploration.ipynb              # Jupyter 交互式分析
```

---

## 配置说明 `config.py`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `RAW_DATA_PATH` | 自动检测 | 数据文件路径，自动扫描 `data/raw/` |
| `CATEGORY_COL` | `None`（自动选） | 用于类别权重分析的列名 |
| `WEIGHT_METHOD` | `"frequency"` | 权重方式：`frequency` 频数 / `proportion` 占比 |
| `MATRIX_ROW_COL` | `None`（自动选） | 矩阵行分类列 |
| `MATRIX_COL_COL` | `None`（自动选） | 矩阵列分类列 |
| `MODEL_TYPE` | `"all"` | `"all"` 全模型对比，或指定单个算法 |
| `TARGET_COL` | `None`（自动选） | 预测目标列 |
| `DO_PREDICTION` | `True` | 是否执行预测建模 |

> **提示**：所有 `None` 的配置项都会自动从数据中智能选择，你也可以手动指定。

---

## 输出示例

运行后会在 `output/` 目录生成 7 个文件：

| 文件 | 说明 |
|------|------|
| `weight_distribution.png` | 类别权重条形图 + 饼图 |
| `weight_matrix_heatmap.png` | 归一化权重矩阵热力图 |
| `correlation_heatmap.png` | 数值特征相关系数热力图 |
| `model_comparison.png` | 全模型 R² / F1 对比排名柱状图 |
| `model_comparison.csv` | 全模型指标排名数据表 |
| `prediction_results.png` | 实际 vs 预测散点图 + 残差分布 |
| `feature_importance.png` | 模型特征重要性排名 |

控制台也会输出完整的分析报告：
```
类别 'category' 的频数权重：
category    count  weight
电子产品     88      88
服装        85      85
...
>>> 权重最大的类别: [电子产品], 权重 = 88
```

---

## 支持的预测模型

`MODEL_TYPE` 可设为以下任意值，或设为 `"all"` 一次性全部对比：

| 模型 | `MODEL_TYPE` | 回归 | 分类 | 说明 |
|------|:-----------:|:----:|:----:|------|
| Random Forest | `random_forest` | ✅ | ✅ | 默认推荐，自带特征重要性 |
| Extra Trees | `extra_trees` | ✅ | ✅ | 随机性更强的树集成 |
| Gradient Boosting | `gradient_boosting` | ✅ | ✅ | 逐步提升，精度高 |
| Decision Tree | `decision_tree` | ✅ | ✅ | 单棵决策树，可解释性强 |
| AdaBoost | `adaboost` | ✅ | ✅ | 自适应增强 |
| Linear / Logistic | `linear` | ✅ | ✅ | 简单快速，强可解释性 |
| Ridge | `ridge` | ✅ | — | L2 正则化线性回归 |
| Lasso | `lasso` | ✅ | — | L1 正则化，自动特征选择 |
| SVM | `svm` | ✅ | ✅ | 支持向量机 |
| KNN | `knn` | ✅ | ✅ | K 近邻 |
| Naive Bayes | `naive_bayes` | — | ✅ | 朴素贝叶斯（仅分类） |
| MLP | `mlp` | ✅ | ✅ | 多层感知器神经网络 |
| XGBoost | `xgboost` | ✅ | ✅ | 需 `pip install xgboost` |
| LightGBM | `lightgbm` | ✅ | ✅ | 需 `pip install lightgbm` |

程序会自动判断任务是分类还是回归，自动进行标准化（SVM/KNN/MLP/线性模型）。

---

## 注意事项

- Excel 文件 (`.xls`) 需要 `xlrd>=2.0.0`
- 中文图表需要系统安装中文字体（如 SimHei / Microsoft YaHei），否则可能显示为方块。程序会自动尝试多个常见字体
- 预测功能默认使用最后一列数值列作为目标列，可在 `config.py` 中修改 `TARGET_COL`
- `MODEL_TYPE = "all"` 时会训练所有可用算法并提供 5 折交叉验证评分排名
