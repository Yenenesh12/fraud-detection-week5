#!/usr/bin/env python3
"""
Simple SHAP Analysis Script for Fraud Detection
Run this directly with: python notebooks/fraud_shap_simple.py
"""

# Import all required libraries
print("Importing libraries...")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

print("✅ All imports successful!")

# Create synthetic fraud data
def create_fraud_data(n_samples=1000, fraud_rate=0.03):
    """Create synthetic fraud detection dataset"""
    np.random.seed(42)
    n_fraud = int(n_samples * fraud_rate)
    n_normal = n_samples - n_fraud
    
    print(f"Creating {n_samples:,} transactions with {fraud_rate*100:.1f}% fraud rate...")
    
    data = []
    
    # Generate normal transactions
    for i in range(n_normal):
        data.append({
            'Amount': max(np.random.lognormal(3, 1), 1),
            'Time_Hour': np.random.choice(range(6, 23)),
            'Location_Risk': np.random.normal(0.2, 0.1),
            'User_Age': max(np.random.normal(35, 12), 18),
            'Account_Age_Days': np.random.exponential(365),
            'Device_Score': max(min(np.random.normal(0.8, 0.1), 1), 0),
            'V1': np.random.normal(0, 1),
            'V2': np.random.normal(0, 1),
            'V3': np.random.normal(0, 1),
            'Class': 0
        })
    
    # Generate fraud transactions
    for i in range(n_fraud):
        data.append({
            'Amount': max(np.random.lognormal(5, 1.5), 1),
            'Time_Hour': np.random.choice(range(24)),
            'Location_Risk': max(min(np.random.normal(0.7, 0.2), 1), 0),
            'User_Age': max(np.random.normal(35, 12), 18),
            'Account_Age_Days': np.random.exponential(200),
            'Device_Score': max(min(np.random.normal(0.3, 0.2), 1), 0),
            'V1': np.random.normal(2, 1),
            'V2': np.random.normal(-1, 1),
            'V3': np.random.normal(1, 1),
            'Class': 1
        })
    
    df = pd.DataFrame(data)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

# Create dataset
print("\n" + "="*50)
print("CREATING DATASET")
print("="*50)
df = create_fraud_data(1000, 0.03)
print(f"Dataset created: {df.shape}")
print(f"Fraud rate: {df['Class'].mean()*100:.1f}%")

# Prepare data
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

print(f"Training set: {X_train_scaled.shape}")
print(f"Test set: {X_test_scaled.shape}")

# Train model
print("\n" + "="*50)
print("TRAINING MODEL")
print("="*50)
rf_model = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)

rf_model.fit(X_train_scaled, y_train)

y_pred = rf_model.predict(X_test_scaled)
y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

auc_roc = roc_auc_score(y_test, y_pred_proba)
auc_pr = average_precision_score(y_test, y_pred_proba)

print(f"AUC-ROC: {auc_roc:.4f}")
print(f"AUC-PR: {auc_pr:.4f}")
print(f"Fraud cases detected: {(y_pred == 1).sum()}/{(y_test == 1).sum()}")

# SHAP Analysis
print("\n" + "="*50)
print("SHAP ANALYSIS")
print("="*50)

print("Initializing SHAP TreeExplainer...")
explainer = shap.TreeExplainer(rf_model)

print("Computing SHAP values...")
shap_sample_size = min(100, len(X_test_scaled))
X_test_sample = X_test_scaled.iloc[:shap_sample_size]
shap_values = explainer.shap_values(X_test_sample)

print(f"SHAP values type: {type(shap_values)}")
if isinstance(shap_values, list):
    print(f"SHAP values list length: {len(shap_values)}")
    print(f"SHAP values[0] shape: {shap_values[0].shape}")
    print(f"SHAP values[1] shape: {shap_values[1].shape}")
    shap_values_fraud = shap_values[1]  # Use fraud class (class 1)
else:
    print(f"SHAP values shape: {shap_values.shape}")
    # For newer versions of SHAP, it might return a 3D array
    if len(shap_values.shape) == 3:
        shap_values_fraud = shap_values[:, :, 1]  # Use fraud class (class 1)
    else:
        shap_values_fraud = shap_values

print(f"✅ SHAP values computed for {shap_sample_size} samples")
print(f"Final SHAP values shape: {shap_values_fraud.shape}")

# Feature importance
mean_abs_shap_values = np.abs(shap_values_fraud).mean(axis=0)
print(f"Mean SHAP values shape: {mean_abs_shap_values.shape}")

mean_abs_shap = pd.DataFrame({
    'feature': X.columns,
    'mean_abs_shap': mean_abs_shap_values
})
mean_abs_shap = mean_abs_shap.sort_values('mean_abs_shap', ascending=False)

print("\n🎯 SHAP Feature Importance Ranking:")
print(mean_abs_shap)

# Built-in importance
feature_importance_rf = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
})
feature_importance_rf = feature_importance_rf.sort_values('importance', ascending=False)

print("\n🔍 Random Forest Feature Importance:")
print(feature_importance_rf)

# Correlation
correlation = feature_importance_rf.set_index('feature')['importance'].corr(
    mean_abs_shap.set_index('feature')['mean_abs_shap']
)

print(f"\n📈 Correlation between RF and SHAP importance: {correlation:.4f}")

if correlation > 0.8:
    print("✅ Strong alignment between built-in and SHAP importance")
elif correlation > 0.6:
    print("🟡 Moderate alignment")
else:
    print("🔴 Weak alignment")

print("\n" + "="*50)
print("ANALYSIS COMPLETE")
print("="*50)
print("✅ SHAP analysis completed successfully!")
print("✅ All libraries are working correctly!")
print("\nYou can now run the Jupyter notebook with confidence.")