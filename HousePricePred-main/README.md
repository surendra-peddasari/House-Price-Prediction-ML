# House Price Prediction — Ames Housing

End-to-end machine learning project for **house price prediction and real-estate analytics** using the Ames, Iowa housing dataset.

The repository is organized as a GitHub-ready project, while the main work remains in a **single Jupyter notebook**, matching the supplied project requirement.

## Project Objective

The project brief asks for:

1. A complete data analysis report.
2. A robust machine-learning algorithm to predict house prices.
3. Analysis of the relationship between house features and price.
4. Practical suggestions for customers based on area, price and other requirements.
5. A comparison of multiple models and a recommendation for production.
6. A report describing challenges and the techniques used to address them.

The project uses `SalePrice` as the target variable and works with the Ames housing features described in the supplied project brief.

## Main Notebook

**`notebooks/House_Price_Prediction.ipynb`**

The notebook is the primary submission and contains the complete workflow:

### 1. Dataset acquisition
- Downloads the project dataset automatically.
- Extracts the archive.
- Locates `train.csv`.
- Loads the data with Pandas.

### 2. Exploratory Data Analysis
- Dataset shape and data types
- Numerical/categorical feature counts
- Duplicate-row check
- Target summary statistics
- `SalePrice` distribution and skewness
- Log-transformed target distribution
- `GrLivArea` vs `SalePrice`
- Missing-value analysis
- Neighborhood price analysis
- Overall quality vs price

### 3. Preprocessing
- Domain-informed outlier removal
- `SalePrice` log transformation using `log1p`
- Identifier removal
- Structural categorical missing-value handling
- Structural numerical missing-value handling
- Neighborhood-based `LotFrontage` imputation
- Median/mode imputation for remaining missing values
- One-hot encoding

### 4. Feature Engineering

The notebook creates:

| Feature | Meaning |
|---|---|
| `TotalSF` | Basement + first-floor + second-floor square footage |
| `TotalBathrooms` | Weighted total of full and half bathrooms |
| `HouseAge` | Year sold minus construction year |
| `YearsSinceRemodel` | Year sold minus remodel year |
| `OverallQual_TotalSF` | Interaction between overall quality and total area |

### 5. Model Benchmarking

Four regression models are evaluated with 5-fold cross-validation:

- Ridge Regression
- Random Forest
- LightGBM
- XGBoost

Metrics:

- Mean CV RMSE on log-transformed price
- Standard deviation of CV RMSE
- Mean CV R²

### 6. Model Comparison

The supplied project notebook reports the following benchmark:

| Rank | Model | Mean CV RMSE (Log) | Std CV RMSE | Mean CV R² |
|---:|---|---:|---:|---:|
| 1 | **Ridge Regression** | **0.1153** | 0.0092 | **0.9163** |
| 2 | XGBoost | 0.1242 | 0.0085 | 0.9026 |
| 3 | LightGBM | 0.1261 | **0.0062** | 0.8998 |
| 4 | Random Forest | 0.1355 | 0.0092 | 0.8837 |

**Interpretation:** Ridge is the reported numerical-performance winner. LightGBM is retained as a strong production-oriented alternative when non-linear relationships and fold-to-fold stability are important.

The notebook recomputes the metrics when executed, so exact values can vary slightly with package versions and execution environment.

## Project Structure

```text
HousePricePred/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── House_Price_Prediction.ipynb
│
├── data/
│   └── README.md
│
├── docs/
│   └── PROJECT_REQUIREMENTS.md
│
└── reports/
    ├── MODEL_COMPARISON.md
    └── CHALLENGES_AND_MITIGATIONS.md
```

## Dataset

The project brief describes the dataset as Ames, Iowa residential housing data with **79 explanatory variables** covering characteristics such as:

- Building class and zoning
- Lot size and configuration
- Neighborhood
- Construction quality and condition
- Year built and remodel year
- Basement characteristics
- Living area
- Bathrooms and bedrooms
- Kitchen quality
- Garage characteristics
- Porches and decks
- Pool and fence information
- Sale month/year
- Sale type and sale condition

The target variable is `SalePrice`.

### Dataset source

The notebook uses the project-provided download location:

`https://d3ilbtxij3aepc.cloudfront.net/projects/CDS-Capstone-Projects/PRCP-1020-HousePricePred.zip`

The raw dataset is **not included in this Git repository**. It is downloaded automatically when the notebook runs.

## Installation

### 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd HousePricePred
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Jupyter

```bash
jupyter notebook
```

Open:

```text
notebooks/House_Price_Prediction.ipynb
```

Run the notebook from top to bottom.

## Requirements

The main dependencies are:

- Python 3.10+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- LightGBM
- Jupyter

See `requirements.txt` for the installable dependency list.

## Data Processing Logic

### Outlier handling

The notebook removes observations satisfying:

```text
GrLivArea > 4000
AND
SalePrice < 300000
```

This is a domain-informed treatment of extreme high-leverage observations rather than a generic statistical deletion rule.

### Missing values

Missing values are treated according to their meaning:

- Features representing the absence of a garage, basement, pool, fence, etc. are assigned explicit `None`/zero values.
- `LotFrontage` uses neighborhood-level median imputation.
- Remaining numerical values use median imputation.
- Remaining categorical values use mode imputation.

### Target transformation

The model target is:

```python
np.log1p(SalePrice)
```

This reduces the effect of extreme price values and provides a more stable regression target.

## Feature/Price Relationship Analysis

The notebook examines the relationship between:

- `OverallQual` and `SalePrice`
- `GrLivArea` and `SalePrice`
- `TotalSF` and price-related behavior
- Neighborhood and median sale price
- Construction/remodel age and price context
- Garage, basement and bathroom characteristics

Correlation is used as an exploratory relationship measure and should not be interpreted as causal evidence.

## Model Selection

### Ridge Regression

Ridge is a strong candidate for this dataset because one-hot encoding produces a high-dimensional feature matrix. L2 regularization helps control coefficient magnitude and reduce overfitting.

### Random Forest

Random Forest provides a non-linear ensemble baseline and can capture feature interactions without requiring a strictly linear relationship.

### LightGBM

LightGBM is a gradient-boosting model designed for efficient tabular learning and is useful when complex non-linear relationships are important.

### XGBoost

XGBoost provides another strong gradient-boosting benchmark and is included to compare a different tree-boosting implementation.

## Challenges and Solutions

| Challenge | Solution |
|---|---|
| Right-skewed target | `log1p(SalePrice)` |
| Structural missing values | Domain-specific `None`/zero treatment |
| Extreme high-leverage properties | Domain-informed outlier trimming |
| High-dimensional categorical data | One-hot encoding |
| Multicollinearity / coefficient instability | Ridge L2 regularization |
| Non-linear relationships | Random Forest, LightGBM and XGBoost |
| Mixed feature scales | RobustScaler |
| Need for actionable interpretation | Neighborhood, quality and size analysis |

A detailed version is available in `reports/CHALLENGES_AND_MITIGATIONS.md`.

## Customer Buying Recommendations

The notebook turns the analysis into practical buyer guidance:

1. Define the maximum budget.
2. Shortlist neighborhoods appropriate to that budget.
3. Compare overall quality and usable floor area.
4. Review construction and remodeling age.
5. Evaluate garage, basement and bathroom configuration.
6. Compare the asking price with model/data-based valuation context.
7. Use the model as a decision-support tool, not a replacement for inspection or professional appraisal.

## Important Reproducibility Note

The benchmark values documented in the project materials are preserved as reported results. The notebook also contains executable code that recomputes the benchmark from the downloaded dataset.

Because preprocessing, package versions, and model implementations can affect results, a fresh execution may produce slightly different values.

## Future Improvements

Potential extensions include:

- Leakage-safe preprocessing with `Pipeline` and `ColumnTransformer`
- Hyperparameter optimization with Optuna
- More rigorous feature selection
- Ensemble/blending of strong models
- SHAP-based explainability
- Geospatial features
- Historical economic and mortgage-rate variables
- Deployment through a REST API
- A small web interface for interactive property valuation

## Academic / Project Context

This repository is based on the supplied `PRCP-1020-HousePricePred` project brief. The brief specifically requires the analysis, model comparison, challenges report and customer recommendations to be completed in a single Jupyter notebook.

## License

No license is added by default because the supplied project materials did not specify a repository license. Add a license before making the repository public if you want to define reuse terms.
