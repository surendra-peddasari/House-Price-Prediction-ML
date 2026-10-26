# Model Comparison Report

## Models

The project benchmarks:

1. Ridge Regression
2. Random Forest
3. LightGBM
4. XGBoost

Evaluation uses 5-fold cross-validation with:

- Mean RMSE on the log-transformed target
- Standard deviation of RMSE
- Mean R²

## Reported Results

| Rank | Model | Mean CV RMSE (Log) | Std CV RMSE | Mean CV R² |
|---:|---|---:|---:|---:|
| 1 | **Ridge Regression** | **0.1153** | 0.0092 | **0.9163** |
| 2 | XGBoost | 0.1242 | 0.0085 | 0.9026 |
| 3 | LightGBM | 0.1261 | **0.0062** | 0.8998 |
| 4 | Random Forest | 0.1355 | 0.0092 | 0.8837 |

## Recommendation

Ridge Regression is the reported winner when the primary objective is minimizing validation RMSE and maximizing R².

LightGBM remains a useful production-oriented alternative when non-linear relationships and cross-validation stability are more important.

The notebook recomputes the results, so exact values should be treated as environment-dependent.
