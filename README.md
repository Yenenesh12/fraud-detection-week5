# Comprehensive Fraud Detection System

A production-ready fraud detection system implementing state-of-the-art machine learning techniques with proper handling of imbalanced data, comprehensive evaluation metrics, and model interpretability.

## 🎯 Overview

This fraud detection pipeline provides a complete solution for detecting fraudulent transactions using advanced machine learning techniques. The system is designed with best practices for imbalanced data, proper evaluation metrics, and production deployment considerations.

### Key Features

✅ **Robust Methodology for Imbalanced Data**
- AUC-PR as primary metric (more appropriate than accuracy for fraud detection)
- Stratified train-test splits to preserve class distribution
- Class weighting and SMOTE for handling imbalance
- Business-focused metrics (fraud detection rate, false alarm rate)

✅ **Comprehensive Model Development**
- Baseline model (Logistic Regression) for interpretability
- Ensemble models (Random Forest, XGBoost, LightGBM)
- Hyperparameter tuning with cross-validation
- Voting ensemble for improved performance

✅ **Production-Ready Features**
- Model persistence and loading
- Comprehensive logging and error handling
- Reproducible results with fixed random seeds
- Modular design for easy customization

✅ **Advanced Analysis & Visualization**
- Feature importance analysis
- SHAP values for model interpretability
- Comprehensive evaluation plots
- Business impact assessment

## 📁 Project Structure

```
fraud-detection/
├── .vscode/                     # VS Code settings
├── .github/                     # GitHub workflows
├── data/                        # Data directory (add to .gitignore)
│   ├── raw/                     # Original datasets
│   └── processed/               # Cleaned and processed data
├── notebooks/                   # Jupyter notebooks
│   ├── __init__.py
│   ├── modeling.ipynb          # Main modeling notebook
│   └── README.md
├── src/                         # Source code
│   ├── __init__.py
│   ├── fraud_detection_pipeline.py      # Core pipeline
│   └── fraud_detection_advanced.py      # Advanced pipeline
├── tests/                       # Unit tests
│   ├── __init__.py
│   └── test_fraud_detection.py
├── models/                      # Saved model artifacts
├── scripts/                     # Executable scripts
│   ├── __init__.py
│   ├── run_fraud_detection.py   # Main execution script
│   └── README.md
├── visualizations/              # Generated plots (created automatically)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── .gitignore                   # Git ignore rules
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd fraud-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Usage

**Option A: Command Line Interface**

```bash
# Run with demo data
python scripts/run_fraud_detection.py --demo

# Run with your dataset
python scripts/run_fraud_detection.py --data data/raw/creditcard.csv

# Run advanced pipeline with visualizations
python scripts/run_fraud_detection.py --data data/raw/creditcard.csv --advanced
```

**Option B: Python API**

```python
from src.fraud_detection_pipeline import FraudDetectionPipeline

# Initialize pipeline
pipeline = FraudDetectionPipeline(random_state=42)

# Run complete pipeline
results = pipeline.run_complete_pipeline('data/raw/creditcard.csv')

# Access results
print(f"Best Model: {results['best_model_name']}")
print(f"AUC-PR: {results['results'][results['best_model_name']]['auc_pr']:.4f}")
```

**Option C: Jupyter Notebook**

```bash
# Start Jupyter
jupyter notebook

# Open notebooks/modeling.ipynb for interactive analysis
```

### 3. Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python tests/test_fraud_detection.py
```

## 📊 Supported Datasets

### Format 1: creditcard.csv (Kaggle Credit Card Fraud Detection)
- **Target variable**: `Class` (0 = normal, 1 = fraud)
- **Features**: V1-V28 (PCA-transformed), Amount, Time
- **Size**: ~284K transactions
- **Imbalance**: ~0.17% fraud rate

### Format 2: Fraud_Data.csv (Custom Format)
- **Target variable**: `class` (0 = normal, 1 = fraud)
- **Features**: Raw transaction features (user_id, purchase_value, etc.)
- **Size**: Variable
- **Imbalance**: Variable

### Synthetic Data (Demo Mode)
- Automatically generated for demonstration
- Configurable fraud rate and sample size
- Realistic transaction patterns

## 🔧 Pipeline Components

### Step 1: Data Preparation
- **Data Loading**: Automatic format detection
- **Exploratory Analysis**: Class distribution, missing values, correlations
- **Feature Preparation**: Missing value handling, scaling
- **Train-Test Split**: Stratified split preserving class distribution

### Step 2: Baseline Model
- **Logistic Regression** with `class_weight='balanced'`
- High interpretability through coefficient analysis
- Fast training and prediction
- Establishes performance baseline

### Step 3: Ensemble Models
- **Random Forest**: Robust to outliers, built-in feature importance
- **XGBoost**: State-of-the-art gradient boosting with scale_pos_weight
- **LightGBM**: Efficient gradient boosting (optional)
- **Voting Ensemble**: Combines top-performing models

### Step 4: Model Evaluation
- **Primary Metric**: AUC-PR (Area Under Precision-Recall Curve)
- **Secondary Metrics**: F1-Score, Precision, Recall, ROC-AUC
- **Business Metrics**: Fraud detection rate, false alarm rate
- **Cross-Validation**: 5-fold stratified for stability assessment

### Step 5: Model Selection
- Multi-criteria selection considering:
  - Performance (AUC-PR, F1-Score)
  - Stability (CV standard deviation)
  - Business requirements
- Detailed justification and trade-off analysis

## 📈 Evaluation Metrics

### Why AUC-PR over Accuracy?

In fraud detection with typical 1-3% fraud rates:
- A model predicting "no fraud" for everything gets 97-99% accuracy
- But catches 0% of actual fraud cases
- **AUC-PR focuses on positive class performance**, making it ideal for imbalanced data

### Key Metrics Explained

- **AUC-PR**: Area under Precision-Recall curve (primary metric)
- **F1-Score**: Harmonic mean of precision and recall
- **Precision**: Accuracy of fraud predictions (minimize false alarms)
- **Recall**: Fraud detection rate (maximize fraud caught)
- **Fraud Detection Rate**: % of actual fraud cases identified
- **False Alarm Rate**: % of legitimate transactions flagged as fraud

## 🎛️ Configuration Options

### Model Parameters

```python
# Customize Random Forest
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}

# Customize XGBoost
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2]
}
```

### Class Imbalance Handling

```python
# Option 1: Class weighting (default)
LogisticRegression(class_weight='balanced')

# Option 2: SMOTE oversampling
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)

# Option 3: Custom weights
class_weight = {0: 1, 1: 10}  # 10x weight for fraud class
```

## 📊 Example Results

```
================================================================================
FRAUD DETECTION PIPELINE REPORT
================================================================================

Dataset Summary:
- Total samples: 284,807
- Features: 30
- Class imbalance: 577.9:1

Models Evaluated:
1. Logistic Regression
2. Random Forest  
3. XGBoost

Best Model: XGBoost
- AUC-PR: 0.8456
- F1-Score: 0.8234
- Precision: 0.8567
- Recall: 0.7923

Business Impact:
- Fraud cases caught: 390 out of 492 (79.3%)
- False alarms: 67 (0.02% of non-fraud)
- Missed fraud: 102 cases

Cross-Validation Stability:
- AUC-PR: 0.8456 ± 0.0234 (Good stability)
- F1-Score: 0.8234 ± 0.0189 (Good stability)
```

## 🔍 Model Interpretability

### Feature Importance
- **Tree-based models**: Built-in feature importance scores
- **Logistic Regression**: Coefficient magnitudes and directions
- **SHAP Analysis**: Model-agnostic explanations (advanced pipeline)

### Business Insights
- Identify most predictive features for fraud
- Understand how feature values influence predictions
- Validate model decisions align with domain knowledge

## 🚀 Production Deployment

### Model Loading and Prediction

```python
import joblib
import pandas as pd

# Load saved model and scaler
model = joblib.load('models/best_model_xgboost.pkl')
scaler = joblib.load('models/scaler.pkl')

# Prediction function
def predict_fraud(transaction_data):
    """
    Predict fraud probability for new transactions
    
    Args:
        transaction_data: DataFrame with transaction features
    
    Returns:
        Array of fraud probabilities
    """
    # Scale features
    scaled_data = scaler.transform(transaction_data)
    
    # Get fraud probability
    fraud_prob = model.predict_proba(scaled_data)[:, 1]
    
    return fraud_prob

# Example usage
new_transaction = pd.DataFrame({
    'Amount': [150.00],
    'V1': [-1.359807],
    'V2': [-0.072781],
    # ... other features
})

fraud_probability = predict_fraud(new_transaction)
risk_level = "HIGH" if fraud_probability[0] > 0.5 else "LOW"
print(f"Fraud Probability: {fraud_probability[0]:.3f} ({risk_level} RISK)")
```

### Real-time Scoring API

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model once at startup
model = joblib.load('models/best_model_xgboost.pkl')
scaler = joblib.load('models/scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get transaction data
        data = request.json
        transaction_df = pd.DataFrame([data])
        
        # Make prediction
        fraud_prob = predict_fraud(transaction_df)[0]
        
        # Return result
        return jsonify({
            'fraud_probability': float(fraud_prob),
            'risk_level': 'HIGH' if fraud_prob > 0.5 else 'LOW',
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## 📚 Best Practices Implemented

### Fraud Detection Methodology
1. **Appropriate Metrics**: AUC-PR over accuracy for imbalanced data
2. **Temporal Considerations**: Proper train-test splits (no data leakage)
3. **Class Imbalance**: Multiple techniques (weighting, SMOTE, cost-sensitive)
4. **Cross-Validation**: Stratified K-fold maintaining class distribution
5. **Feature Engineering**: Domain-specific transformations
6. **Model Interpretability**: Business understanding and compliance

### Software Engineering
1. **Modular Design**: Separate classes for different functionalities
2. **Error Handling**: Comprehensive exception handling
3. **Testing**: Unit tests for all major components
4. **Documentation**: Detailed docstrings and comments
5. **Reproducibility**: Fixed random seeds throughout
6. **Logging**: Detailed progress and decision logging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run tests (`python -m pytest tests/`)
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

# Run linting
black src/ tests/ scripts/

# Type checking (optional)
mypy src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support & Troubleshooting

### Common Issues

1. **"No dataset found" error**
   - Use `--demo` flag for synthetic data
   - Place your dataset in `data/raw/` directory
   - Ensure correct file format (CSV with proper target column)

2. **Memory issues with large datasets**
   - Use data sampling in pipeline initialization
   - Consider feature selection for high-dimensional data
   - Use advanced pipeline with memory optimization

3. **Poor model performance**
   - Check class distribution (severe imbalance may need custom handling)
   - Verify feature quality and relevance
   - Adjust hyperparameter tuning ranges

4. **Import errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility (3.8+)
   - Verify virtual environment activation

### Getting Help

- Check the [Issues](../../issues) page for known problems
- Review the example notebooks in `notebooks/`
- Run the test suite to verify installation: `python -m pytest tests/`

## 📊 Performance Benchmarks

Typical performance on standard datasets:

| Dataset | Samples | Features | Fraud Rate | Best AUC-PR | Best F1 | Training Time |
|---------|---------|----------|------------|-------------|---------|---------------|
| Credit Card | 284K | 30 | 0.17% | 0.85+ | 0.80+ | ~5 min |
| Synthetic | 5K | 10 | 3.0% | 0.75+ | 0.70+ | ~30 sec |

*Performance may vary based on hardware and data characteristics*

---

**Note**: This system is designed for educational and research purposes. For production fraud detection systems, additional considerations include real-time processing, regulatory compliance, model monitoring, and integration with existing infrastructure.