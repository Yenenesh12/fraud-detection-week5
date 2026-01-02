#!/usr/bin/env python3
"""
Main execution script for fraud detection pipeline.

This script provides a command-line interface to run the fraud detection pipeline
with various options and configurations.

Usage:
    python scripts/run_fraud_detection.py --data data/raw/creditcard.csv
    python scripts/run_fraud_detection.py --data data/raw/Fraud_Data.csv --advanced
    python scripts/run_fraud_detection.py --demo

Author: AI Assistant
Date: 2026-01-02
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from fraud_detection_pipeline import FraudDetectionPipeline
from fraud_detection_advanced import AdvancedFraudDetectionPipeline
import pandas as pd
import numpy as np


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
        time_hour = np.random.choice(range(6, 23), p=[0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
        
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
        time_hour = np.random.choice(range(24), p=[0.08]*5 + [0.02]*12 + [0.08]*7)
        
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


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Fraud Detection Pipeline')
    parser.add_argument('--data', type=str, help='Path to the dataset CSV file')
    parser.add_argument('--advanced', action='store_true', help='Use advanced pipeline with visualizations')
    parser.add_argument('--demo', action='store_true', help='Run demo with synthetic data')
    parser.add_argument('--random-state', type=int, default=42, help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    print("="*80)
    print("FRAUD DETECTION PIPELINE")
    print("="*80)
    
    # Determine data source
    if args.demo:
        print("Running in DEMO mode with synthetic data...")
        synthetic_df = create_synthetic_data(
            n_samples=5000,
            fraud_rate=0.03,
            random_state=args.random_state
        )
        data_path = 'data/raw/synthetic_fraud_data.csv'
        synthetic_df.to_csv(data_path, index=False)
        print(f"Synthetic data saved to: {data_path}")
        
    elif args.data:
        data_path = args.data
        if not os.path.exists(data_path):
            print(f"ERROR: Data file not found: {data_path}")
            return 1
            
    else:
        # Auto-detect available datasets
        possible_paths = [
            'data/raw/creditcard.csv',
            'data/raw/Fraud_Data.csv',
            'creditcard.csv',
            'Fraud_Data.csv'
        ]
        
        data_path = None
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                break
        
        if data_path is None:
            print("ERROR: No dataset found!")
            print("Please provide a dataset using one of these options:")
            print("  --data path/to/your/dataset.csv")
            print("  --demo (to use synthetic data)")
            print("\nOr place your dataset in one of these locations:")
            for path in possible_paths:
                print(f"  {path}")
            return 1
    
    print(f"Using dataset: {data_path}")
    
    # Initialize and run pipeline
    try:
        if args.advanced:
            print("Using ADVANCED pipeline with visualizations and SHAP analysis...")
            pipeline = AdvancedFraudDetectionPipeline(random_state=args.random_state)
        else:
            print("Using STANDARD pipeline...")
            pipeline = FraudDetectionPipeline(random_state=args.random_state)
        
        # Run the pipeline
        results = pipeline.run_complete_pipeline(data_path)
        
        print("\n" + "="*80)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        # Display summary
        best_model = results['best_model_name']
        best_results = results['results'][best_model]
        
        print(f"\nSUMMARY:")
        print(f"Best Model: {best_model}")
        print(f"AUC-PR: {best_results['auc_pr']:.4f}")
        print(f"F1-Score: {best_results['f1_score']:.4f}")
        print(f"Precision: {best_results['precision']:.4f}")
        print(f"Recall: {best_results['recall']:.4f}")
        
        print(f"\nOutput Files:")
        print("- models/: Trained models and results")
        if args.advanced:
            print("- visualizations/: Analysis plots and charts")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: Pipeline execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)