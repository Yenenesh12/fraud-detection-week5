# Comprehensive Fraud Detection Pipeline

A production-ready fraud detection system implementing state-of-the-art machine learning techniques with proper handling of imbalanced data, comprehensive evaluation metrics, and model interpretability.

## 🎯 Overview

This fraud detection pipeline provides:

- **Robust Data Preparation**: Proper handling of missing values, outliers, and feature scaling
- **Baseline & Ensemble Models**: Logistic Regression, Random Forest, XGBoost, and ensemble methods
- **Imbalanced Data Handling**: SMOTE, class weighting, and proper evaluation metrics
- **Cross-Validation**: Stratified K-fold for reliable performance assessment
- **Model Interpretability**: Feature importance analysis and SHAP values
- **Production Ready**: Model persistence, comprehensive logging, and evaluation

## 📊 Key Features

✅ **Proper Methodology for Imbalanced Data**
- AUC-PR as primary metric (more appropriate than accuracy for fraud detection)
- Stratified train-test splits to preserve class distribution
- Class weighting and SMOTE for handling imbalance
- Business-focused metrics (fraud detection rate, false alarm rate)

✅ **Comprehensive Model Evaluation**
- Multiple algorithms: Logistic Regression, Random Forest, XGBoost
- Hyperparameter tuning with cross-validation
- Stability assessment through CV standard deviation
- Confusion matrices and precision-recall curves

✅ **Interpretability & Explainability**
- Feature importance analysis for tree-based models
- Logistic regression coefficients interpretation
- SHAP analysis for model explanations (optional)
- Business impact metrics

✅ **Production Readiness**
- Model persistence with joblib
- Comprehensive logging and error handling
- Reproducible results with fixed random seeds
- Modular design for easy customization

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements_fraud.txt
```

### Basic Usage

1. **With Real Data** (creditcard.csv or Fraud_Data.csv):
```python
from fraud_detection_pipeline import FraudDetectionPipeline

# Initialize pipeline
pipeline = FraudDetectionPipeline(random_state=42)

# Run complete pipeline
results = pipeline.run_complete_pipeline('path/to/your/data.csv')
```

2. **Demo with Synthetic Data**:
```python
python demo_fraud_detection.py
```

3. **Advanced Pipeline with Visualizations**:
```python
from fraud_detection_advanced import AdvancedFraudDetectionPipeline

pipeline = AdvancedFraudDetectionPipeline(random_state=42)
results = pipeline.run_complete_pipeline('path/to/your/data.csv')
```

## 📁 Project Structure

```
fraud-detection/
├── fraud_detection_pipeline.py      # Core pipeline implementation
├── fraud_detection_advanced.py      # Enhanced pipeline with visualizations
├── demo_fraud_detection.py          # Demo with synthetic data
├── requirements_fraud.txt           # Python dependencies
├── README_FRAUD_DETECTION.md       # This file
├── models/                          # Saved models and results
│   ├── best_model_*.pkl            # Best performing model
│   ├── scaler.pkl                  # Feature scaler
│   ├── model_comparison.csv        # Detailed comparison table
│   ├── results.json                # Complete results
│   └── metadata.json               # Pipeline metadata
└── visualizations/                 # Generated plots and charts
    ├── eda_overview.png            # Exploratory data analysis
    ├── model_comparison.png        # Model performance comparison
    ├── *_evaluation.png            # Individual model evaluations
    └── shap_*.png                  # SHAP interpretability plots
```

## 🔧 Dataset Requirements

### Supported Formats

1. **creditcard.csv** (Kaggle Credit Card Fraud Detection):
   - Target variable: `Class` (0 = normal, 1 = fraud)
   - Features: PCA-transformed features (V1-V28), Amount, Time

2. **Fraud_Data.csv** (Custom fraud dataset):
   - Target variable: `class` (0 = normal, 1 = fraud)
   - Features: Raw transaction features

### Data Assumptions

- CSV format with header row
- Target variable encoded as 0 (normal) and 1 (fraud)
- Missing values handled automatically
- Numerical features scaled automatically
- Categorical features encoded appropriately

## 📈 Pipeline Steps

### Step 1: Data Preparation
- Load and validate dataset
- Exploratory data analysis with class distribution
- Handle missing values and outliers
- Feature scaling for numerical variables

### Step 2: Baseline Model Development
- **Logistic Regression** with `class_weight='balanced'`
- Interpretable coefficients for feature importance
- Fast training and prediction
- Good baseline for comparison

### Step 3: Ensemble Model Development
- **Random Forest** with hyperparameter tuning
- **XGBoost** with scale_pos_weight for imbalance
- **LightGBM** (optional, if installed)
- **Voting Ensemble** of top models

### Step 4: Cross-Validation
- **Stratified K-Fold** (k=5) to preserve class distribution
- Multiple metrics: AUC-PR, F1-Score, Precision, Recall
- Stability assessment through standard deviation analysis

### Step 5: Model Comparison & Selection
- Comprehensive comparison table with all metrics
- Multi-criteria selection considering:
  - AUC-PR (primary metric for imbalanced data)
  - F1-Score (balance of precision and recall)
  - Cross-validation stability
  - Business requirements (interpretability vs. performance)

### Step 6: Model Interpretability
- Feature importance analysis
- SHAP values for model explanations
- Business impact assessment

## 📊 Evaluation Metrics

### Primary Metrics (Imbalanced Data Focus)
- **AUC-PR (Area Under Precision-Recall Curve)**: Primary metric
- **F1-Score**: Harmonic mean of precision and recall
- **Precision**: Accuracy of fraud predictions (minimize false alarms)
- **Recall**: Fraud detection rate (maximize fraud caught)

### Business Metrics
- **Fraud Detection Rate**: % of actual fraud cases caught
- **False Alarm Rate**: % of normal transactions flagged as fraud
- **Cost-Benefit Analysis**: Business impact of different thresholds

### Why Not Accuracy?
In fraud detection with 2% fraud rate:
- A model predicting "no fraud" for everything gets 98% accuracy
- But catches 0% of fraud cases
- AUC-PR focuses on positive class performance

## 🎛️ Customization Options

### Model Parameters
```python
# Customize models
pipeline = FraudDetectionPipeline(random_state=42)

# Modify hyperparameter grids in the source code
# Random Forest parameters
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}

# XGBoost parameters
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2]
}
```

### Evaluation Metrics
```python
# Add custom metrics in cross_validate scoring
scoring = ['average_precision', 'f1', 'precision', 'recall', 'roc_auc']
```

### Class Imbalance Handling
```python
# Options for handling imbalance:
# 1. Class weighting (default)
LogisticRegression(class_weight='balanced')

# 2. SMOTE oversampling
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)

# 3. Custom class weights
class_weight = {0: 1, 1: 10}  # 10x weight for fraud class
```

## 📋 Example Output

```
================================================================================
FRAUD DETECTION PIPELINE INITIALIZED
================================================================================
Random State: 42
Timestamp: 2026-01-02 10:30:45
================================================================================

============================================================
STEP 1: DATA PREPARATION
============================================================
Loading data from: data/raw/creditcard.csv
Dataset shape: (284807, 31)
Memory usage: 65.32 MB
Target variable: 'Class' (creditcard.csv format)

--- EXPLORATORY DATA ANALYSIS ---
Dataset Info:
Shape: (284807, 31)
Features: 30
Samples: 284,807

Class Distribution:
Non-fraud (0): 284,315 (99.83%)
Fraud (1): 492 (0.17%)
Imbalance Ratio: 577.9:1

============================================================
STEP 2: BASELINE MODEL DEVELOPMENT
============================================================

--- LOGISTIC REGRESSION BASELINE ---
Training Logistic Regression with class_weight='balanced'...

--- EVALUATING LOGISTIC REGRESSION (BASELINE) ---
Performance Metrics:
AUC-PR (Primary): 0.7234
F1-Score: 0.7456
Precision: 0.8123
Recall: 0.6891
ROC-AUC: 0.9456

Business Interpretation:
- Correctly identified fraud cases: 339 out of 492 (68.9%)
- False alarms: 84 out of 423 predictions (19.9%)
- Missed fraud cases: 153 (potential losses)

============================================================
STEP 3: ENSEMBLE MODEL DEVELOPMENT
============================================================

--- RANDOM FOREST ---
Best parameters: {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 2}
Best CV AUC-PR: 0.8567

--- EVALUATING RANDOM FOREST ---
Performance Metrics:
AUC-PR (Primary): 0.8234
F1-Score: 0.8123
Precision: 0.8456
Recall: 0.7801
ROC-AUC: 0.9678

--- XGBOOST ---
Best parameters: {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.1}
Best CV AUC-PR: 0.8789

--- EVALUATING XGBOOST ---
Performance Metrics:
AUC-PR (Primary): 0.8456
F1-Score: 0.8234
Precision: 0.8567
Recall: 0.7923
ROC-AUC: 0.9723

============================================================
STEP 4: CROSS-VALIDATION
============================================================

--- XGBOOST CROSS-VALIDATION ---
AUC-PR: 0.8456 ± 0.0234
F1-Score: 0.8234 ± 0.0189
Stability assessment: Good stability

============================================================
STEP 5: MODEL COMPARISON AND SELECTION
============================================================

MODEL COMPARISON TABLE:
================================================================================
              Model  Test_AUC_PR  Test_F1  Test_Precision  Test_Recall  CV_AUC_PR_Mean  CV_AUC_PR_Std
  Logistic Regression       0.7234   0.7456          0.8123       0.6891          0.7123         0.0345
       Random Forest       0.8234   0.8123          0.8456       0.7801          0.8156         0.0267
             XGBoost       0.8456   0.8234          0.8567       0.7923          0.8456         0.0234

Selected Best Model: XGBoost

Justification for selecting XGBoost:
- AUC-PR: 0.8456 (highest)
- F1-Score: 0.8234
- Cross-validation stability: ±0.0234
- Improvement over baseline: 16.9%

================================================================================
FINAL FRAUD DETECTION PIPELINE REPORT
================================================================================

Best Model: XGBoost
- AUC-PR: 0.8456
- F1-Score: 0.8234
- Precision: 0.8567
- Recall: 0.7923

Business Impact:
- Fraud cases caught: 390 out of 492 (79.3%)
- False alarms: 67 (0.02% of non-fraud)
- Missed fraud: 102 cases

Recommendations:
1. Deploy the selected model for real-time fraud detection
2. Monitor model performance and retrain periodically
3. Implement feedback loop for continuous improvement
4. Consider ensemble of top models for production
```

## 🔍 Model Interpretability

### Feature Importance
- **Tree-based models**: Built-in feature importance scores
- **Logistic Regression**: Coefficient magnitudes and signs
- **SHAP Analysis**: Model-agnostic explanations

### Business Insights
- Which features are most predictive of fraud
- How feature values influence fraud probability
- Model decision boundaries and thresholds

## 🚀 Production Deployment

### Model Persistence
```python
import joblib

# Load saved model
model = joblib.load('models/best_model_xgboost.pkl')
scaler = joblib.load('models/scaler.pkl')

# Make predictions
def predict_fraud(transaction_data):
    scaled_data = scaler.transform(transaction_data)
    fraud_probability = model.predict_proba(scaled_data)[:, 1]
    return fraud_probability
```

### Real-time Scoring
```python
# Example real-time prediction
transaction = {
    'Amount': 150.00,
    'V1': -1.359807,
    'V2': -0.072781,
    # ... other features
}

fraud_prob = predict_fraud([list(transaction.values())])
if fraud_prob[0] > 0.5:
    print("HIGH RISK: Manual review required")
else:
    print("LOW RISK: Transaction approved")
```

## 📚 References & Best Practices

### Fraud Detection Best Practices
1. **Use appropriate metrics**: AUC-PR over accuracy for imbalanced data
2. **Preserve temporal order**: Don't use future data to predict past
3. **Handle class imbalance**: SMOTE, class weighting, or cost-sensitive learning
4. **Cross-validation**: Stratified K-fold to maintain class distribution
5. **Feature engineering**: Domain knowledge is crucial
6. **Model interpretability**: Regulatory compliance and business understanding

### Academic References
- Chawla, N. V. et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique
- Davis, J. & Goadrich, M. (2006). The relationship between Precision-Recall and ROC curves
- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Update documentation
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the console output for detailed error messages
2. Verify data format matches expected structure
3. Ensure all dependencies are installed
4. Review the example outputs above

---

**Note**: This pipeline is designed for educational and research purposes. For production fraud detection systems, additional considerations include real-time processing, model monitoring, regulatory compliance, and integration with existing systems.