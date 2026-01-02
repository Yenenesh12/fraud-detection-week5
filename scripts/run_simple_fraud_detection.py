#!/usr/bin/env python3
"""
Simplified fraud detection script without SMOTE dependency.
This version uses class weighting instead of SMOTE for handling imbalanced data.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

def create_synthetic_data(n_samples=5000, fraud_rate=0.03, random_state=42):
    """Create synthetic fraud detection dataset for demonstration"""
    np.random.seed(random_state)
    
    print(f"Creating synthetic dataset with {n_samples:,} samples and {fraud_rate*100:.1f}% fraud rate...")
    
    # Calculate number of fraud and non-fraud cases
    n_fraud = int(n_samples * fraud_rate)
    n_normal = n_samples - n_fraud
    
    data = []
    
    # Generate normal transactions
    for i in range(n_normal):
        amount = np.random.lognormal(mean=3, sigma=1)
        time_hour = np.random.choice(range(6, 23))
        
        data.append({
            'Amount': max(amount, 1),
            'Time_Hour': time_hour,
            'Location_Risk': np.random.normal(0.2, 0.1),
            'User_Age': max(np.random.normal(35, 12), 18),
            'Account_Age_Days': np.random.exponential(365),
            'Transactions_Last_Hour': np.random.poisson(0.5),
            'Transactions_Last_Day': np.random.poisson(5),
            'Device_Score': max(min(np.random.normal(0.8, 0.1), 1), 0),
            'V1': np.random.normal(0, 1),
            'V2': np.random.normal(0, 1),
            'V3': np.random.normal(0, 1),
            'V4': np.random.normal(0, 1),
            'V5': np.random.normal(0, 1),
            'Class': 0
        })
    
    # Generate fraud transactions
    for i in range(n_fraud):
        amount = np.random.lognormal(mean=5, sigma=1.5)
        time_hour = np.random.choice(range(24))
        
        data.append({
            'Amount': max(amount, 1),
            'Time_Hour': time_hour,
            'Location_Risk': max(min(np.random.normal(0.7, 0.2), 1), 0),
            'User_Age': max(np.random.normal(35, 12), 18),
            'Account_Age_Days': np.random.exponential(200),
            'Transactions_Last_Hour': np.random.poisson(2),
            'Transactions_Last_Day': np.random.poisson(15),
            'Device_Score': max(min(np.random.normal(0.3, 0.2), 1), 0),
            'V1': np.random.normal(2, 1),
            'V2': np.random.normal(-1, 1),
            'V3': np.random.normal(1, 1),
            'V4': np.random.normal(-2, 1),
            'V5': np.random.normal(1.5, 1),
            'Class': 1
        })
    
    # Create DataFrame and shuffle
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return df

def run_fraud_detection_pipeline(data_path=None, demo=True):
    """Run the complete fraud detection pipeline"""
    
    print("="*80)
    print("SIMPLIFIED FRAUD DETECTION PIPELINE")
    print("="*80)
    
    # Load or create data
    if demo:
        print("Running in DEMO mode with synthetic data...")
        df = create_synthetic_data(n_samples=5000, fraud_rate=0.03, random_state=42)
    else:
        print(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)
    
    print(f"Dataset shape: {df.shape}")
    
    # Detect target column
    if 'Class' in df.columns:
        target_col = 'Class'
    elif 'class' in df.columns:
        target_col = 'class'
    else:
        print("ERROR: Could not find target column ('Class' or 'class')")
        return None
    
    # Basic EDA
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    print(f"Target column: {target_col}")
    class_counts = df[target_col].value_counts()
    print(f"Class distribution:")
    print(f"Non-fraud (0): {class_counts[0]:,} ({class_counts[0]/len(df)*100:.2f}%)")
    print(f"Fraud (1): {class_counts[1]:,} ({class_counts[1]/len(df)*100:.2f}%)")
    print(f"Imbalance ratio: {class_counts[0]/class_counts[1]:.1f}:1")
    
    # Prepare features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Handle non-numeric columns
    X = X.select_dtypes(include=[np.number])
    
    print(f"Features used: {list(X.columns)}")
    print(f"Number of features: {X.shape[1]}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]:,} samples")
    print(f"Test set: {X_test.shape[0]:,} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n" + "="*60)
    print("MODEL TRAINING AND EVALUATION")
    print("="*60)
    
    models = {}
    results = {}
    
    # 1. Logistic Regression (Baseline)
    print("\n--- LOGISTIC REGRESSION (BASELINE) ---")
    lr = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    models['Logistic Regression'] = lr
    
    # Evaluate
    y_pred_lr = lr.predict(X_test_scaled)
    y_pred_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]
    
    results['Logistic Regression'] = {
        'auc_pr': average_precision_score(y_test, y_pred_proba_lr),
        'f1_score': f1_score(y_test, y_pred_lr),
        'precision': precision_score(y_test, y_pred_lr),
        'recall': recall_score(y_test, y_pred_lr),
        'roc_auc': roc_auc_score(y_test, y_pred_proba_lr)
    }
    
    print(f"AUC-PR: {results['Logistic Regression']['auc_pr']:.4f}")
    print(f"F1-Score: {results['Logistic Regression']['f1_score']:.4f}")
    print(f"Precision: {results['Logistic Regression']['precision']:.4f}")
    print(f"Recall: {results['Logistic Regression']['recall']:.4f}")
    
    # 2. Random Forest
    print("\n--- RANDOM FOREST ---")
    rf = RandomForestClassifier(
        n_estimators=100, 
        class_weight='balanced', 
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    models['Random Forest'] = rf
    
    # Evaluate
    y_pred_rf = rf.predict(X_test_scaled)
    y_pred_proba_rf = rf.predict_proba(X_test_scaled)[:, 1]
    
    results['Random Forest'] = {
        'auc_pr': average_precision_score(y_test, y_pred_proba_rf),
        'f1_score': f1_score(y_test, y_pred_rf),
        'precision': precision_score(y_test, y_pred_rf),
        'recall': recall_score(y_test, y_pred_rf),
        'roc_auc': roc_auc_score(y_test, y_pred_proba_rf)
    }
    
    print(f"AUC-PR: {results['Random Forest']['auc_pr']:.4f}")
    print(f"F1-Score: {results['Random Forest']['f1_score']:.4f}")
    print(f"Precision: {results['Random Forest']['precision']:.4f}")
    print(f"Recall: {results['Random Forest']['recall']:.4f}")
    
    # Cross-validation for stability
    print("\n--- CROSS-VALIDATION ---")
    cv_scores_lr = cross_val_score(lr, X_train_scaled, y_train, cv=5, scoring='average_precision')
    cv_scores_rf = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring='average_precision')
    
    print(f"Logistic Regression CV AUC-PR: {cv_scores_lr.mean():.4f} ± {cv_scores_lr.std():.4f}")
    print(f"Random Forest CV AUC-PR: {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}")
    
    # Model comparison
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df.round(4)
    print(comparison_df)
    
    # Select best model
    best_model_name = comparison_df['auc_pr'].idxmax()
    best_model = models[best_model_name]
    
    print(f"\nBest Model: {best_model_name}")
    print(f"AUC-PR: {results[best_model_name]['auc_pr']:.4f}")
    
    # Business impact analysis
    print("\n" + "="*60)
    print("BUSINESS IMPACT ANALYSIS")
    print("="*60)
    
    if best_model_name == 'Logistic Regression':
        y_pred_best = y_pred_lr
    else:
        y_pred_best = y_pred_rf
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_best)
    tn, fp, fn, tp = cm.ravel()
    
    fraud_detection_rate = tp / (tp + fn) * 100
    false_alarm_rate = fp / (fp + tn) * 100
    
    print(f"Fraud cases caught: {tp} out of {tp + fn} ({fraud_detection_rate:.1f}%)")
    print(f"False alarms: {fp} out of {fp + tn} ({false_alarm_rate:.3f}%)")
    print(f"Missed fraud cases: {fn}")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    return {
        'models': models,
        'results': results,
        'best_model_name': best_model_name,
        'scaler': scaler,
        'feature_names': list(X.columns)
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Simplified Fraud Detection Pipeline')
    parser.add_argument('--data', type=str, help='Path to the dataset CSV file')
    parser.add_argument('--demo', action='store_true', help='Run demo with synthetic data')
    
    args = parser.parse_args()
    
    if args.demo:
        results = run_fraud_detection_pipeline(demo=True)
    elif args.data:
        if os.path.exists(args.data):
            results = run_fraud_detection_pipeline(data_path=args.data, demo=False)
        else:
            print(f"ERROR: File not found: {args.data}")
    else:
        print("Please specify --demo or --data path/to/dataset.csv")
        print("Example: python scripts/run_simple_fraud_detection.py --demo")