# 定量数据分析 Skill

以**定量分析**为核心的 Python 数据分析工具包：按类别统计 → 计算权重 → 构建矩阵 → 绘制权重分布图 → 算法预测。

适配绝大部分 CSV / XLSX 文件，自动检测编码、分隔符、列类型，零配置即可运行。

---

## 核心流程

```
加载数据 → 清洗 → 基础统计 → 类别权重 → 权重矩阵 → 可视化 → 预测建模
```

1. **数据加载** — 自动识别 .csv / .xlsx / .xls，探测编码和分隔符
2. **数据清洗** — 去重、缺失值填充（均值/中位数/众数）、异常值处理
3. **基础描述性统计** — 摘要统计、缺失值报告、相关系数矩阵
4. **类别权重计算** — 按类别统计频数/占比，找出权重最大的类别
5. **权重矩阵构建** — 二维交叉矩阵 + 归一化
6. **权重分布图** — 条形图 + 饼图 + 矩阵热力图
7. **预测建模** — Random Forest / Linear / XGBoost，自动判断分类/回归

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
│   ├── quantitative_analysis.py       # ★ 定量分析核心：权重计算、矩阵构建
│   ├── visualize.py                   # ★ 可视化：权重分布图、热力图、预测图
│   └── predict.py                     # ★ 预测建模：RF/Linear/XGBoost
├── output/                            # 所有图表和结果输出
│   ├── weight_distribution.png        # 类别权重分布图
│   ├── weight_matrix_heatmap.png      # 权重矩阵热力图
│   ├── correlation_heatmap.png        # 相关系数热力图
│   ├── prediction_results.png         # 预测结果：实际 vs 预测
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
| `MODEL_TYPE` | `"random_forest"` | 预测模型：`random_forest` / `linear` / `xgboost` |
| `TARGET_COL` | `None`（自动选） | 预测目标列 |
| `DO_PREDICTION` | `True` | 是否执行预测建模 |

> **提示**：所有 `None` 的配置项都会自动从数据中智能选择，你也可以手动指定。

---

## 输出示例

运行后会在 `output/` 目录生成：

- **权重分布图** — 条形图展示各类别频数/占比 + 饼图展示百分比
- **权重矩阵热力图** — 二维交叉矩阵的可视化
- **相关系数热力图** — 数值特征间的相关性
- **预测结果图** — 实际值 vs 预测值散点图 + 残差分布直方图
- **特征重要性图** — 模型判断哪些特征对预测最重要

控制台也会输出完整的分析报告：
```
类别 'category' 的频数权重：
category    count  weight
电子产品     88      88
服装        85      85
食品        83      83
家居        83      83
运动户外     82      82
图书        79      79
>>> 权重最大的类别: [电子产品], 权重 = 88
```

---

## 支持的模型

| 模型 | 分类 | 回归 | 说明 |
|------|:----:|:----:|------|
| Random Forest | ✅ | ✅ | 默认推荐，自带特征重要性 |
| Linear / Logistic | ✅ | ✅ | 简单快速，可解释性强 |
| XGBoost | ✅ | ✅ | 需 `pip install xgboost`，性能更优 |

程序会自动判断任务是分类还是回归。

---

## 注意事项

- Excel 文件 (`.xls`) 需要 `xlrd>=2.0.0`
- 中文图表需要系统安装中文字体（如 SimHei / Microsoft YaHei），否则可能显示为方块。程序会自动尝试多个常见字体
- 预测功能默认使用最后一列数值列作为目标列，可在 `config.py` 中修改 `TARGET_COL`
