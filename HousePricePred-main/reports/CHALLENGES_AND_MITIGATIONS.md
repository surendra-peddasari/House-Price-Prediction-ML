# Challenges and Mitigations

| Challenge | Risk | Mitigation |
|---|---|---|
| Target skewness | Large prices can dominate the regression error scale | Apply `log1p(SalePrice)` |
| Structural missing values | Missing can represent feature absence | Use `None` for structural categorical fields and zero for structural numeric fields |
| Extreme high-leverage homes | Unusual large/low-priced homes can distort the model | Remove observations matching `GrLivArea > 4000` and `SalePrice < 300000` |
| High-dimensional categorical variables | One-hot encoding expands the feature space | Use one-hot encoding and regularized Ridge regression |
| Non-linear feature interactions | Linear models cannot represent every relationship | Benchmark Random Forest, LightGBM and XGBoost |
| Mixed feature scales | Features use different numerical ranges | Apply RobustScaler |
| Customer interpretability | Predictions alone are not actionable | Analyze neighborhood, quality, size, age and amenities |
