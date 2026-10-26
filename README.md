# House Price Prediction ML

An end-to-end machine learning project for **house price prediction and residential real-estate analysis** using the Ames, Iowa housing dataset.

The project combines exploratory data analysis, domain-aware preprocessing, feature engineering, regression model benchmarking, cross-validation, and practical home-buying recommendations. The complete analytical workflow is implemented in a **single Jupyter notebook**, as required by the project brief.

---

## Project Overview

House prices depend on many interacting factors such as location, overall construction quality, living area, basement and garage facilities, age of the property, and sale conditions.

This project uses the Ames Housing dataset to build a regression system that predicts `SalePrice` and investigates which property characteristics are most strongly associated with price.

### Project objectives

- Perform a complete exploratory data analysis of the housing dataset.
- Understand the relationship between house features and `SalePrice`.
- Clean and preprocess numerical and categorical variables.
- Handle structural missing values using domain knowledge.
- Engineer useful real-estate features.
- Build and compare multiple machine-learning regression models.
- Use 5-fold cross-validation to evaluate model performance.
- Recommend a model based on predictive performance and operational considerations.
- Convert the analysis into practical guidance for potential home buyers.
- Document challenges, assumptions, limitations, and future improvements.

---

## Dataset

The project uses the **Ames, Iowa residential housing dataset** described in the supplied project specification. The dataset contains **79 explanatory variables** describing almost every aspect of residential properties.

The target variable is:

```text
SalePrice
```

which represents the property's final sale price in dollars.

### Important features

Examples of the available attributes include:

- `MSSubClass` — building class
- `MSZoning` — zoning classification
- `LotArea` — lot size
- `Neighborhood` — physical location within Ames
- `OverallQual` — overall material and finish quality
- `OverallCond` — overall condition
- `YearBuilt` — original construction year
- `YearRemodAdd` — remodeling year
- `GrLivArea` — above-ground living area
- `TotalBsmtSF` — total basement area
- `FullBath` / `HalfBath` — bathroom counts
- `BedroomAbvGr` — bedrooms above basement level
- `KitchenQual` — kitchen quality
- `GarageCars` — garage capacity
- `GarageArea` — garage area
- `WoodDeckSF` / `OpenPorchSF` — outdoor areas
- `PoolArea` — pool area
- `MoSold` / `YrSold` — sale month and year
- `SaleType` / `SaleCondition` — sale information

The original project specification contains the complete attribute description.

### Dataset source

The notebook downloads the project dataset programmatically from the project-provided source:

```text
https://d3ilbtxij3aepc.cloudfront.net/projects/CDS-Capstone-Projects/PRCP-1020-HousePricePred.zip
```

The raw dataset is intentionally not required to be committed to GitHub. This keeps the repository lightweight and allows the notebook to reproduce the data-loading process.

---

## Repository Structure

```text
House-Price-Prediction-ML/
│
├── README.md
│
└── HousePricePred-main/
    │
    ├── requirements.txt
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

> **GitHub recommendation:** If you want the cleanest repository presentation, move the contents of `HousePricePred-main/` to the repository root so that `README.md`, `notebooks/`, `data/`, `docs/`, `reports/`, and `requirements.txt` appear directly at the top level.

---

## Main Notebook

The main deliverable is:

```text
notebooks/House_Price_Prediction.ipynb
```

The notebook contains the complete project workflow in one place:

```text
Dataset Download & Loading
        ↓
Data Inspection
        ↓
Exploratory Data Analysis
        ↓
Outlier Analysis
        ↓
Missing-Value Analysis
        ↓
Preprocessing
        ↓
Feature Engineering
        ↓
5-Fold Cross-Validation
        ↓
Model Comparison
        ↓
Production Recommendation
        ↓
Customer Buying Recommendations
        ↓
Challenges & Mitigations
        ↓
Conclusion & Future Scope
```

---

## Exploratory Data Analysis

The notebook investigates the structure and behavior of the housing data, including:

- Dataset dimensions and feature types.
- Numerical and categorical feature counts.
- Target-variable distribution.
- Target skewness.
- Log-transformed `SalePrice` distribution.
- `GrLivArea` versus `SalePrice`.
- Missing-value patterns.
- Neighborhood-level price behavior.
- Overall quality and price relationships.

### Target transformation

`SalePrice` is strongly right-skewed. The notebook therefore creates a log-transformed target:

```python
np.log1p(SalePrice)
```

This makes the target distribution more suitable for regression modeling and reduces the influence of extreme prices.

---

## Data Cleaning & Preprocessing

### Outlier treatment

The notebook removes properties satisfying:

```text
GrLivArea > 4000
AND
SalePrice < 300000
```

These observations are treated as high-leverage anomalies that can distort the regression relationship between living area and price.

### Missing values

Missing values are interpreted using domain knowledge rather than applying one generic imputation rule.

For example, an absent garage or basement can represent a real property characteristic rather than an unknown value.

The notebook therefore uses:

- Explicit categorical values such as `None` for structural absence.
- Zero values where a missing numerical measurement represents no such feature.
- Neighborhood-based median imputation for `LotFrontage`.
- Median imputation for remaining numerical variables.
- Mode imputation for remaining categorical variables.

### Encoding

Categorical variables are converted into machine-learning-ready numerical features using one-hot encoding.

### Scaling

The model benchmarking stage applies `RobustScaler`, which is less sensitive to residual outliers than standard mean/variance scaling.

---

## Feature Engineering

The notebook creates domain-informed features to represent broader property characteristics.

| Feature | Description |
|---|---|
| `TotalSF` | Combined basement, first-floor, and second-floor square footage |
| `TotalBathrooms` | Weighted combination of full and half bathrooms |
| `HouseAge` | Year sold minus construction year |
| `YearsSinceRemodel` | Year sold minus remodeling year |
| `OverallQual_TotalSF` | Interaction between overall quality and total area |

These engineered variables attempt to capture property characteristics that are more directly meaningful for valuation than isolated raw columns.

---

## Machine Learning Models

Four regression algorithms are benchmarked:

### 1. Ridge Regression

Ridge Regression applies L2 regularization and is particularly useful after one-hot encoding because the resulting feature space is high-dimensional and can contain correlated predictors.

### 2. Random Forest

Random Forest provides a non-linear ensemble baseline and can model interactions without requiring a linear relationship between every feature and price.

### 3. LightGBM

LightGBM is a gradient-boosting algorithm designed for efficient tabular machine learning and can capture complex non-linear relationships.

### 4. XGBoost

XGBoost is another gradient-boosting approach used as a strong benchmark against the other regression models.

---

## Model Evaluation

The models are evaluated using **5-fold cross-validation** with a fixed random state of `42`.

The notebook reports:

- Mean CV RMSE on the log-transformed target.
- Standard deviation of CV RMSE.
- Mean CV R².

### Recorded benchmark

The notebook contains the following benchmark results:

| Rank | Model | Mean CV RMSE (Log) | Std CV RMSE | Mean CV R² |
|---:|---|---:|---:|---:|
| 1 | **Ridge Regression** | **0.1153** | 0.0092 | **0.9163** |
| 2 | XGBoost | 0.1242 | 0.0085 | 0.9026 |
| 3 | LightGBM | 0.1261 | **0.0062** | 0.8998 |
| 4 | Random Forest | 0.1355 | 0.0092 | 0.8837 |

These are the results recorded in the notebook. A fresh execution can produce slightly different results if the environment, package versions, or dataset state changes.

### Model interpretation

**Ridge Regression** is the numerical-performance winner in the recorded benchmark because it has the lowest CV RMSE and highest mean CV R².

**LightGBM** has the lowest recorded RMSE standard deviation, making it the most stable of the four models across the folds in this experiment.

The notebook therefore presents a dual perspective:

- **Ridge Regression:** strongest recorded predictive metric performance.
- **LightGBM:** strong non-linear alternative with the lowest recorded fold-to-fold RMSE variation.

---

## Key Price Drivers

The notebook's analysis highlights several important factors associated with residential sale prices:

- `OverallQual` — overall material and finish quality.
- `TotalSF` — combined usable/constructed floor area.
- `Neighborhood` — geographic location within Ames.
- `GrLivArea` — above-grade living area.
- Basement characteristics.
- Garage capacity and area.
- Construction and remodeling age.
- Bathroom configuration.

Feature relationships should be interpreted as **associations useful for prediction**, not proof of causation.

---

## Customer Home-Buying Recommendations

The project translates the analysis into practical buyer guidance.

### 1. Consider neighborhood first

The notebook identifies premium neighborhoods such as:

- Northridge Heights (`NridgHt`)
- Stone Brook (`StoneBr`)
- Northridge (`NoRidge`)

It also identifies lower-entry-price neighborhoods such as:

- Old Town
- Edwards
- Sawyer

### 2. Prioritize quality and usable area

`OverallQual` and total floor area are highlighted as major valuation signals. Buyers should evaluate the combination of structural quality and usable space rather than focusing only on bedroom count.

### 3. Evaluate functional amenities

Basement finish, garage capacity, bathrooms, and other functional features can materially affect property value and usefulness.

### 4. Consider age and renovation history

Construction year and remodeling year provide useful context for maintenance requirements and expected property condition.

### 5. Use predictions as decision support

A machine-learning valuation should support a buyer's decision rather than replace professional inspection, appraisal, legal checks, or local market research.

---

## Challenges & Technical Solutions

| Challenge | Approach |
|---|---|
| Right-skewed `SalePrice` | `log1p` target transformation |
| Structural missing values | Domain-specific categorical/zero imputation |
| High-leverage properties | Domain-informed outlier removal |
| Many categorical variables | One-hot encoding |
| Correlated/high-dimensional features | Ridge L2 regularization |
| Non-linear relationships | Random Forest, LightGBM and XGBoost |
| Residual outliers in feature space | RobustScaler |
| Need for business interpretation | Neighborhood, quality and size analysis |

Detailed documentation is available in:

```text
reports/CHALLENGES_AND_MITIGATIONS.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/surendra-peddasari/House-Price-Prediction-ML.git
cd House-Price-Prediction-ML
```

If the repository still contains the nested `HousePricePred-main` directory, enter it before installing dependencies:

```bash
cd HousePricePred-main
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

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
notebooks/House_Price_Prediction.ipynb
```

Run the notebook from top to bottom.

---

## Requirements

The project uses Python and the following major libraries:

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- LightGBM
- XGBoost
- Jupyter Notebook

The exact dependency list is provided in:

```text
requirements.txt
```

---

## Reproducibility

For more consistent results:

- Use the same Python/package environment.
- Keep the cross-validation random state fixed.
- Download the same dataset version.
- Run the notebook from the beginning rather than executing isolated cells.
- Record package versions when publishing benchmark results.

The current notebook uses `random_state=42` for the cross-validation setup and model configurations where applicable.

---

## Project Deliverables

The project addresses the three main tasks from the supplied project specification:

### Task 1 — Data Analysis

A complete analysis of the housing data, including distributions, missing values, outliers, relationships, and feature behavior.

### Task 2 — Machine Learning & Relationships

Multiple regression algorithms are trained and compared to predict house prices, while the notebook investigates how property characteristics relate to `SalePrice`.

### Task 3 — Customer Recommendations

The notebook translates the data analysis into practical home-buying suggestions based on neighborhood, price-related features, property quality, area, age, and amenities.

Additional project requirements for **model comparison** and **challenges faced** are also included in the notebook and supporting reports.

---

## Limitations

This project is a machine-learning analysis of historical Ames housing data and has several limitations:

- The dataset represents a specific market and historical period.
- The model should not be assumed to generalize directly to other cities or countries.
- The model does not automatically account for current interest rates, inflation, or changing market conditions.
- Predictions are dependent on the quality and completeness of the input features.
- Machine-learning predictions should not replace professional property appraisal or inspection.

---

## Future Improvements

Potential next steps include:

- Hyperparameter optimization using Optuna or another tuning framework.
- Leakage-safe preprocessing using `Pipeline` and `ColumnTransformer`.
- Model blending or stacking.
- SHAP-based model explainability.
- Geospatial features such as distances to schools, transport, and city centers.
- Historical economic and mortgage-rate variables.
- A REST API for automated valuation.
- A web interface for interactive house-price prediction.
- Model monitoring and automated retraining.
- A dedicated held-out test set for final unbiased evaluation.

---

## Project Status

**Status:** Completed academic machine-learning project / portfolio project.

The repository is structured around the Jupyter notebook as the primary executable project deliverable, with supporting documentation and reports for reproducibility and review.

---

## Author

**Surendra Peddasari**

GitHub: [surendra-peddasari](https://github.com/surendra-peddasari)

---

## License

No license has been specified for this repository. If you intend to make the project openly reusable, consider adding an appropriate open-source license such as MIT.
