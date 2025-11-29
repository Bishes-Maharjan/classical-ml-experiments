# King County House Price Prediction

## Comprehensive Machine Learning Model Comparison

**Author**: Bishes Maharjan  
**Date**: November 2025

A rigorous comparison of 9 regression models for predicting house prices using the King County housing dataset, featuring extensive preprocessing, cross-validation, and performance analysis across multiple dataset variations.

---

## 📋 Executive Summary

This project presents a comprehensive comparison of multiple machine learning regression models for house price prediction. The analysis evaluated 9 different models across two dataset variations:

- **Log-transformed feature-engineered dataset** (33 columns)
- **Original raw dataset** (49 columns)

The study employed cross-validation, learning curves, and validation curves to evaluate model performance and generalization capabilities.

**Best Performing Model**: Gradient Boosting achieved **84.38% R²** on the original dataset.

---

## 📊 Dataset Overview

### Dataset Characteristics

- **Size**: 560,219 property sales records
- **Sample Used**: 50,000 records (randomly sampled for computational efficiency)
- **Features**:
  - Processed dataset: 33 columns
  - Original dataset: 49 columns
- **Target Variable**: Sale price (log-transformed vs. raw values)

### Key Features

- **Geographic**: Latitude, city, zoning
- **Property Characteristics**: Square footage, bedrooms, bathrooms, stories
- **Quality Metrics**: Grade, condition
- **Special Features**: Waterfront, views, renovations
- **Market Information**: Submarket classification

---

## 🔧 Data Preprocessing Pipeline

### Objective

Prepare large-scale real estate data for regression tasks through cleaning, transformation, and feature reduction while preserving predictive power.

### Preprocessing Steps

#### 1. Data Cleaning

- Checked and handled null values
- Dropped irrelevant columns: `Unnamed: 0`, `subdivision` (high null values)
- Removed non-numeric attributes except `submarket` for one-hot encoding

#### 2. Multicollinearity Detection

- Used correlation analysis to identify highly correlated features
- **Dropped features with correlation ≥ 0.8** (except perfect correlation = 1.0)
- Applied one-hot encoding with `drop_first=True` to prevent multicollinearity

#### 3. Feature Transformation

**Target Transformation**:

- Converted `sale_price` → `sale_price_log` using `np.log1p()`
- Purpose: Stabilize variance and reduce skewness

**Numeric Feature Transformation**:

- Identified skewed numeric variables
- Applied log transformation to approximate normality
- Dropped `longitude-log` due to high null values

#### 4. Feature Selection

- Calculated correlation with target variable
- **Dropped features with correlation < 0.1** to target
- Removed non-predictive columns (IDs, dates, warnings, status indicators)

#### 5. Final Dataset

- **Reduced from 49 to 33 columns**
- Saved as separate CSV for comparative analysis
- Enabled testing impact of preprocessing on different model types

### Computational Optimization

- Used **sampling (50,000 records)** instead of full dataset (560k rows)
- Reason: KFolds drastically increased hardware requirements and training time
- Trade-off: Maintained statistical validity while achieving practical runtime

---

## 🤖 Models Evaluated

### Tree-Based Models (3)

1. **Decision Tree Regressor**

   - `max_depth=30`
   - `min_samples_split=50`
   - `min_samples_leaf=20`

2. **Random Forest Regressor**

   - `n_estimators=200`
   - `max_features='sqrt'`
   - `bootstrap=True`
   - `n_jobs=-1`

3. **Gradient Boosting Regressor**
   - `n_estimators=200`
   - `learning_rate=0.2`
   - `max_depth=3` (performed worse with more depth)
   - `subsample=0.8`

### Linear Models (4)

4. **Linear Regression**

   - StandardScaler applied

5. **Ridge Regression** (L2 regularization)

   - `alpha=1.0`

6. **Lasso Regression** (L1 regularization)

   - `alpha=0.01`
   - `max_iter=10000`

7. **ElasticNet Regression** (L1 + L2)
   - `alpha=0.01`
   - `l1_ratio=0.5`

### Polynomial Models (2)

8. **Polynomial Regression** (degree 2)

9. **Polynomial Ridge Regression** (degree 2)

**Note**: Polynomial Lasso and ElasticNet were excluded due to device limitations and excessive training time.

---

## 📈 Results

### Cross-Validation Performance (5-Fold KFold)

#### Log-Transformed Dataset

| Rank | Model                         | R² Mean | R² Std |
| ---- | ----------------------------- | ------- | ------ |
| 1    | Gradient Boosting             | 0.6127  | 0.0059 |
| 2    | Polynomial Regression (deg 2) | 0.6022  | 0.0050 |
| 3    | Polynomial Ridge (deg 2)      | 0.6022  | 0.0050 |
| 4    | Random Forest                 | 0.5831  | 0.0067 |
| 5    | Linear Regression             | 0.5524  | 0.0061 |
| 6    | Ridge Regression              | 0.5524  | 0.0061 |
| 7    | ElasticNet Regression         | 0.5483  | 0.0058 |
| 8    | Lasso Regression              | 0.5412  | 0.0058 |
| 9    | Decision Tree                 | 0.5329  | 0.0079 |

#### Original Dataset

| Rank | Model                         | R² Mean    | R² Std |
| ---- | ----------------------------- | ---------- | ------ |
| 🏆 1 | **Gradient Boosting**         | **0.8438** | 0.0261 |
| 2    | Decision Tree                 | 0.7316     | 0.0188 |
| 3    | Random Forest                 | 0.6341     | 0.0276 |
| 4    | Polynomial Ridge (deg 2)      | 0.6134     | 0.0254 |
| 5    | Polynomial Regression (deg 2) | 0.6130     | 0.0255 |
| 6    | Linear Regression             | 0.5390     | 0.0233 |
| 7    | Ridge Regression              | 0.5390     | 0.0233 |
| 8    | ElasticNet Regression         | 0.5390     | 0.0234 |
| 9    | Lasso Regression              | 0.5390     | 0.0233 |

---

## 🔍 Key Findings

### Model Performance Insights

1. **Gradient Boosting Dominance**

   - Achieved highest performance on both datasets
   - Exceptional performance on original dataset (R² = 0.8438)
   - 84.38% of variance explained

2. **Dataset Impact**

   - Original dataset yielded better performance for tree-based models
   - Linear models showed similar performance across both datasets
   - Preprocessing doesn't always improve results

3. **Polynomial Benefits**

   - Second-degree polynomial features significantly improved linear models
   - Nearly matched ensemble methods on log-transformed dataset
   - However, suffered from numerical instability issues

4. **Regularization Effects**
   - Ridge and Lasso showed minimal performance impact
   - Suggests features were already well-conditioned
   - ElasticNet provided no advantage over simpler approaches

---

## 📉 Learning Curve Analysis

### Key Patterns Observed

#### Tree-Based Models

**Random Forest**:

- ✅ Excellent generalization with converging training/validation curves
- Training: ~0.58-0.59, Validation: ~0.54
- Minimal overfitting, performance stabilizes after ~3,000 samples
- **Interpretation**: Well-balanced model

**Decision Tree**:

- ⚠️ Clear overfitting patterns, especially on original dataset
- Original: Training ~0.70 vs Validation ~0.60
- Large gap indicates memorization of training data
- **Interpretation**: Ensemble methods needed

**Gradient Boosting**:

- ⚠️ Concerning overfitting on derived dataset
- Derived: Training drops 0.85→0.70, validation flat at ~0.58
- Original: Excellent with Training ~0.92, Validation ~0.78
- **Interpretation**: Model complexity too high for log-transformed features

#### Linear Models

**Linear Regression**:

- ✅ Ideal learning behavior with good generalization
- Both curves converge around 0.55
- **Interpretation**: Model reached capacity; more data won't help significantly

**Ridge & Lasso Regression**:

- Nearly identical to Linear Regression
- Confirms regularization has minimal impact
- **Interpretation**: Features well-conditioned, no multicollinearity issues

**ElasticNet Regression**:

- Similar pattern to Linear and Ridge
- Converges around 0.54-0.55
- **Interpretation**: Combined L1+L2 offers no advantage

#### Polynomial Models

**Polynomial Regression (deg 2)**:

- ❌ Catastrophic instability on derived dataset
- Extreme negative R² values (reaching -20)
- **Interpretation**: Severe numerical instability on log-transformed data

**Polynomial Ridge (deg 2)**:

- ⚠️ Severe overfitting on original dataset initially
- R² drops from ~1 to near 0 as training size increases
- **Interpretation**: Extreme complexity requires substantial data

---

## 📊 Validation Curve Insights

### Hyperparameter Sensitivity

- **Decision Trees**: Performance peaked around depth 20-25, then plateaued
- **Random Forest**: Performance improved steadily up to 200-250 estimators
- **Gradient Boosting**: Optimal around 200-250 estimators

**Note**: Validation curves for polynomial models not generated due to computational constraints.

---

## 💡 Critical Insights & Lessons Learned

### 1. Simpler is Sometimes Better

- Original dataset consistently outperformed log-transformed features
- More feature engineering ≠ better performance
- Tree-based algorithms showed strong preference for raw data

### 2. Polynomial Transformations Problematic

- Created severe numerical instability
- Computational issues rather than performance improvements
- Highlights importance of empirical validation

### 3. Preprocessing Strategy Matters

- Must match preprocessing to algorithm type
- Linear models: benefit from normalization and scaling
- Tree models: often prefer raw features

### 4. Regularization Not Always Necessary

- Ridge, Lasso, ElasticNet showed minimal impact
- Indicates well-conditioned feature space
- Avoid unnecessary complexity

### 5. Computational Trade-offs

- Gradient Boosting: best performance but high computational cost
- Random Forest: practical alternative with reasonable performance
- Decision: depends on resource constraints vs. accuracy requirements

---

## 🔮 Future Improvements

### Model Enhancement

- Hyperparameter tuning using Grid Search or Bayesian Optimization
- Ensemble stacking combining top models
- Deep learning approaches (Neural Networks)
- XGBoost or LightGBM for efficiency

### Data Enhancement

- Use full dataset (560k rows) with distributed computing
- Advanced feature engineering (interaction terms, domain knowledge)
- Handle outliers more systematically
- Time-based features if temporal data available

### Analysis Extension

- Feature importance analysis
- SHAP values for model interpretability
- Residual analysis for error patterns
- Model deployment pipeline

---

## 📁 Repository Structure

```
.
├── logic/                       # Training notebooks and scripts
│   ├── model_training.ipynb
│   └── model_preprocessing.ipynb
├── documentation/               # Results, reports, and PDFs
├── dataset/                     # Original and processed datasets
│   ├── kingcountysales.csv
│   └── processed_data.csv (33 columns)
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
Jupyter Notebook
```

### Installation

````bash
# Clone repository
git clone <repository-url>
cd multi-model

### Required Libraries

```bash
pip install numpy pandas scikit-learn matplotlib seaborn tqdm
````

### Running the Analysis

1. **Data Preprocessing**:

```bash
jupyter notebook Training/DataProcessing.ipynb
```

2. **Model Training**:

```bash
jupyter notebook Training/ModelTraining.ipynb
```

---

## 🛠️ Technologies Used

- **Python**: Core programming language
- **Pandas & NumPy**: Data manipulation
- **Scikit-learn**: Machine learning algorithms
- **Matplotlib & Seaborn**: Visualization
- **tqdm**: Progress visualization
- **Jupyter Notebook**: Interactive development

---

## 📊 Evaluation Framework

- **Cross-Validation**: 5-fold KFold with shuffling
- **Primary Metric**: R² (coefficient of determination)
- **Sample Size**: 50,000 records
- **Random State**: 42 (reproducibility)

---

## 🎯 Conclusion

This comprehensive analysis demonstrates that:

1. **Model choice significantly impacts performance** - Gradient Boosting achieved 84.38% R² but requires substantial computational resources

2. **Preprocessing isn't always beneficial** - Original dataset outperformed engineered features for most models

3. **Complex transformations can backfire** - Polynomial features created numerical instability

4. **Empirical validation is essential** - Assumptions about preprocessing must be tested

5. **Practical trade-offs matter** - Random Forest offers good balance between performance and computational efficiency

The findings emphasize the need for systematic experimentation rather than assuming more complex approaches will yield better results.

---

## 📧 Contact

**Bishes Maharjan**

For questions or collaboration, please open an issue in this repository.

---

## 📄 License

[Specify your license]

---

## 🙏 Acknowledgments

- King County Open Data Portal for dataset
- Scikit-learn documentation and community
- Python data science ecosystem

---

**Note**: This project demonstrates advanced machine learning practices including systematic model comparison, learning curve analysis, and critical evaluation of preprocessing strategies.
