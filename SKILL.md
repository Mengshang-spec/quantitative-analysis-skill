---
name: quantitative-analysis
description: >
  Quantitative data analysis with category weight computation, matrix building, and ML prediction.
  Load CSV/XLSX files, clean data, compute category frequencies/weights, build normalized weight
  matrices, generate weight distribution charts (bar+pie+heatmap), and run ML prediction models
  (Random Forest/Linear/XGBoost). Use when the user needs to analyze tabular data, compute category
  weights, build cross-tabulation matrices, visualize weight distributions, or predict outcomes
  from structured data. Triggers: quantitative analysis, data analysis, category weight, weight
  matrix, data visualization, data prediction, CSV analysis, Excel analysis, 定量分析, 权重计算,
  权重矩阵, 数据预测.
---

# Quantitative Analysis Skill

Quantitative data analysis pipeline: load -> clean -> analyze -> weight computation -> matrix -> visualization -> prediction.

## Quick Start

Run the main pipeline with default auto-detection:

```bash
cd <skill-directory>
python main.py
```

To use with a specific data file, place CSV/XLSX files in `data/raw/` or set `RAW_DATA_PATH` in `config.py`.

## Configuration

Edit `config.py` to customize:

- `RAW_DATA_PATH` --- auto-detected from `data/raw/`, or set explicitly
- `CATEGORY_COL` --- column for category weight analysis (None = auto-detect)
- `WEIGHT_METHOD` --- `"frequency"` (count) or `"proportion"` (percentage)
- `MATRIX_ROW_COL` / `MATRIX_COL_COL` --- cross-tabulation dimensions (None = auto-detect)
- `TARGET_COL` --- prediction target column (None = auto-detect last numeric column)
- `MODEL_TYPE` --- `"random_forest"` / `"linear"` / `"xgboost"`
- `DO_PREDICTION` --- `True` to include ML prediction step

## Pipeline Steps

1. **Load** (`src/data_loader.py`) --- Auto-detect CSV/XLSX, encoding, separator
2. **Clean** (`src/data_cleaner.py`) --- Deduplicate, fill missing, remove outliers
3. **Analyze** (`src/analysis.py`) --- Summary stats, correlation, column type detection
4. **Weight** (`src/quantitative_analysis.py`) --- Category frequency/weight, cross-tab matrix, normalization, pairwise weights
5. **Visualize** (`src/visualize.py`) --- Bar+pies charts, heatmaps, prediction scatter plots
6. **Predict** (`src/predict.py`) --- Auto-encode categoricals, train RF/Linear/XGBoost, evaluate, feature importance

## Output

All charts saved to `output/`:
- `weight_distribution.png` --- Category weight bar chart + pie chart
- `weight_matrix_heatmap.png` --- Normalized cross-tab heatmap
- `correlation_heatmap.png` --- Numeric feature correlations
- `prediction_results.png` --- Actual vs predicted + residual distribution
- `feature_importance.png` --- Model feature importance ranking

## Interactive Use

```python
from src.data_loader import load_data
from src.quantitative_analysis import category_frequency, build_weight_matrix
from src.visualize import plot_weight_distribution, plot_weight_matrix_heatmap

df = load_data("data/raw/myfile.csv")
weights = category_frequency(df, "category_col", weight_method="frequency")
plot_weight_distribution(weights, "category_col", save_path="output/my_weights.png")
```

## Notes

- Run with: `D:\miniconda\envs\myenv\python.exe main.py` (or the active Python environment)
- Chinese fonts auto-detected; falls back gracefully if none found
- If no data file exists, a sample dataset is generated automatically
