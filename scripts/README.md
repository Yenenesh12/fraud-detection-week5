# Scripts Directory

This directory contains executable scripts for running the fraud detection pipeline.

## Available Scripts

### `run_fraud_detection.py`

Main execution script for the fraud detection pipeline with command-line interface.

**Usage:**

```bash
# Run with your own dataset
python scripts/run_fraud_detection.py --data data/raw/creditcard.csv

# Run with advanced pipeline (includes visualizations and SHAP)
python scripts/run_fraud_detection.py --data data/raw/Fraud_Data.csv --advanced

# Run demo with synthetic data
python scripts/run_fraud_detection.py --demo

# Run demo with advanced pipeline
python scripts/run_fraud_detection.py --demo --advanced

# Specify custom random state
python scripts/run_fraud_detection.py --demo --random-state 123
```

**Arguments:**

- `--data`: Path to your dataset CSV file
- `--advanced`: Use the advanced pipeline with visualizations and SHAP analysis
- `--demo`: Generate and use synthetic data for demonstration
- `--random-state`: Set random state for reproducibility (default: 42)

**Examples:**

1. **Quick Demo:**
   ```bash
   python scripts/run_fraud_detection.py --demo
   ```

2. **Full Analysis with Real Data:**
   ```bash
   python scripts/run_fraud_detection.py --data data/raw/creditcard.csv --advanced
   ```

3. **Standard Pipeline:**
   ```bash
   python scripts/run_fraud_detection.py --data data/raw/Fraud_Data.csv
   ```

## Output

The scripts will create the following directories and files:

- `models/`: Trained models, scalers, and results
- `visualizations/`: Plots and charts (advanced pipeline only)
- `data/processed/`: Processed datasets

## Requirements

Make sure you have installed all dependencies:

```bash
pip install -r requirements.txt
```

## Supported Data Formats

The pipeline supports two main dataset formats:

1. **creditcard.csv** (Kaggle Credit Card Fraud Detection):
   - Target: `Class` (0=normal, 1=fraud)
   - Features: V1-V28 (PCA), Amount, Time

2. **Fraud_Data.csv** (Custom format):
   - Target: `class` (0=normal, 1=fraud)
   - Features: Raw transaction features

The pipeline automatically detects the format and adjusts accordingly.