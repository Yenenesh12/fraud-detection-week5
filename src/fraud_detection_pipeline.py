"""
Comprehensive Fraud Detection Pipeline
=====================================

This script implements a complete fraud detection pipeline following best practices:
- Robust data preparation with proper handling of imbalanced data
- Baseline and ensemble model development
- Comprehensive evaluation with appropriate metrics for fraud detection
- Cross-validation for model stability assessment
- Model comparison and selection with clear justification

Author: AI Assistant
Date: 2026-01-02
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, roc_auc_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

class FraudDetectionPipeline:
    """
    Complete fraud detection pipeline with proper methodology for imbalanced data.
    
    This class implements:
    1. Data preparation with EDA
    2. Baseline model (Logistic Regression)
    3. Ensemble models (Random Forest, XGBoost)
    4. Cross-validation
    5. Model comparison and selection
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.cv_results = {}
        self.best_model = None
        
        # Set up directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        
        print("="*80)
        print("FRAUD DETECTION PIPELINE INITIALIZED")
        print("="*80)
        print(f"Random State: {self.random_state}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
    
    def load_and_explore_data(self, data_path):
        """
        Step 1: Data Preparation
        Load dataset and perform initial exploration
        """
        print("\n" + "="*60)
        print("STEP 1: DATA PREPARATION")
        print("="*60)
        
        # Load data
        print(f"Loading data from: {data_path}")
        self.df = pd.read_csv(data_path)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Identify target variable
        if 'Class' in self.df.columns:
            self.target_col = 'Class'
            print("Target variable: 'Class' (creditcard.csv format)")
        elif 'class' in self.df.columns:
            self.target_col = 'class'
            print("Target variable: 'class' (Fraud_Data.csv format)")
        else:
            raise ValueError("Target variable not found. Expected 'Class' or 'class'")
        
        # Basic EDA
        self._perform_eda()
        
        return self.df
    
    def _perform_eda(self):
        """Perform exploratory data analysis"""
        print("\n--- EXPLORATORY DATA ANALYSIS ---")
        
        # Dataset info
        print(f"\nDataset Info:")
        print(f"Shape: {self.df.shape}")
        print(f"Features: {self.df.shape[1] - 1}")
        print(f"Samples: {self.df.shape[0]:,}")
        
        # Class distribution
        class_counts = self.df[self.target_col].value_counts()
        class_props = self.df[self.target_col].value_counts(normalize=True)
        
        print(f"\nClass Distribution:")
        print(f"Non-fraud (0): {class_counts[0]:,} ({class_props[0]*100:.2f}%)")
        print(f"Fraud (1): {class_counts[1]:,} ({class_props[1]*100:.2f}%)")
        print(f"Imbalance Ratio: {class_counts[0]/class_counts[1]:.1f}:1")
        
        # Missing values
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print(f"\nMissing Values:")
            print(missing[missing > 0])
        else:
            print("\nNo missing values found")
        
        # Data types
        print(f"\nData Types:")
        print(self.df.dtypes.value_counts())
        
        # Basic statistics for numerical features
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        print(f"\nNumerical Features: {len(numerical_cols)}")
        
        if len(numerical_cols) <= 10:  # Show stats if not too many features
            print("\nBasic Statistics:")
            print(self.df[numerical_cols].describe())
    
    def prepare_features(self):
        """
        Prepare features and handle missing values/outliers
        """
        print("\n--- FEATURE PREPARATION ---")
        
        # Separate features and target
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        
        # Handle missing values if any
        if X.isnull().sum().sum() > 0:
            print("Handling missing values...")
            # For numerical features, use median
            numerical_cols = X.select_dtypes(include=[np.number]).columns
            X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].median())
            
            # For categorical features, use mode or 'Unknown'
            categorical_cols = X.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown')
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        print(f"Features prepared: {len(self.feature_names)}")
        
        return X, y
    
    def split_data(self, X, y):
        """
        Step 2: Stratified train-test split
        """
        print("\n--- DATA SPLITTING ---")
        
        # Stratified split to preserve class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=self.random_state,
            stratify=y,  # This is crucial for imbalanced data
            shuffle=True
        )
        
        # Document the split
        print(f"Training set: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
        print(f"Test set: {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
        
        print(f"\nClass distribution in training set:")
        train_counts = y_train.value_counts()
        train_props = y_train.value_counts(normalize=True)
        print(f"Non-fraud: {train_counts[0]:,} ({train_props[0]*100:.2f}%)")
        print(f"Fraud: {train_counts[1]:,} ({train_props[1]*100:.2f}%)")
        
        print(f"\nClass distribution in test set:")
        test_counts = y_test.value_counts()
        test_props = y_test.value_counts(normalize=True)
        print(f"Non-fraud: {test_counts[0]:,} ({test_props[0]*100:.2f}%)")
        print(f"Fraud: {test_counts[1]:,} ({test_props[1]*100:.2f}%)")
        
        print(f"\nWhy stratification is used:")
        print("- Preserves the original class distribution in both train and test sets")
        print("- Prevents bias where test set might have different fraud rates")
        print("- Ensures reliable evaluation metrics")
        print("- Critical for imbalanced datasets like fraud detection")
        
        # Store splits
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self):
        """Scale numerical features"""
        print("\n--- FEATURE SCALING ---")
        
        # Identify numerical columns
        numerical_cols = self.X_train.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 0:
            self.scaler = StandardScaler()
            
            # Fit on training data only
            self.X_train_scaled = self.X_train.copy()
            self.X_test_scaled = self.X_test.copy()
            
            self.X_train_scaled[numerical_cols] = self.scaler.fit_transform(self.X_train[numerical_cols])
            self.X_test_scaled[numerical_cols] = self.scaler.transform(self.X_test[numerical_cols])
            
            print(f"Scaled {len(numerical_cols)} numerical features")
        else:
            self.X_train_scaled = self.X_train
            self.X_test_scaled = self.X_test
            print("No numerical features to scale")
    
    def train_baseline_model(self):
        """
        Step 3: Baseline Model Development (Logistic Regression)
        """
        print("\n" + "="*60)
        print("STEP 2: BASELINE MODEL DEVELOPMENT")
        print("="*60)
        
        print("\n--- LOGISTIC REGRESSION BASELINE ---")
        print("Justification for Logistic Regression as baseline:")
        print("- High interpretability (coefficients show feature importance)")
        print("- Fast training and prediction")
        print("- Good performance on linearly separable data")
        print("- Provides probability estimates")
        print("- Handles class imbalance well with class_weight='balanced'")
        
        # Train with balanced class weights to handle imbalance
        self.baseline_model = LogisticRegression(
            class_weight='balanced',  # Automatically adjusts for class imbalance
            random_state=self.random_state,
            max_iter=1000
        )
        
        print(f"\nTraining Logistic Regression with class_weight='balanced'...")
        print("This automatically sets class weights inversely proportional to class frequencies")
        
        # Fit the model
        self.baseline_model.fit(self.X_train_scaled, self.y_train)
        
        # Evaluate
        baseline_results = self._evaluate_model(
            self.baseline_model, 
            self.X_test_scaled, 
            self.y_test, 
            "Logistic Regression (Baseline)"
        )
        
        self.models['Logistic Regression'] = self.baseline_model
        self.results['Logistic Regression'] = baseline_results
        
        return self.baseline_model
    
    def train_ensemble_models(self):
        """
        Step 4: Ensemble Model Development
        """
        print("\n" + "="*60)
        print("STEP 3: ENSEMBLE MODEL DEVELOPMENT")
        print("="*60)
        
        # Random Forest
        self._train_random_forest()
        
        # XGBoost
        self._train_xgboost()
    
    def _train_random_forest(self):
        """Train Random Forest with hyperparameter tuning"""
        print("\n--- RANDOM FOREST ---")
        print("Justification for Random Forest:")
        print("- Handles mixed data types well")
        print("- Built-in feature importance")
        print("- Robust to outliers")
        print("- Good performance on tabular data")
        print("- Less prone to overfitting than single trees")
        
        # Calculate scale_pos_weight for class imbalance
        neg_count = (self.y_train == 0).sum()
        pos_count = (self.y_train == 1).sum()
        
        print(f"\nClass imbalance handling:")
        print(f"Using class_weight='balanced' parameter")
        print(f"This sets weight for class 0: {pos_count/neg_count:.3f}")
        print(f"This sets weight for class 1: {neg_count/pos_count:.3f}")
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        print(f"\nPerforming hyperparameter tuning...")
        print(f"Parameter grid: {param_grid}")
        
        rf_base = RandomForestClassifier(
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Use RandomizedSearchCV for efficiency (limit to 10 combinations)
        rf_search = RandomizedSearchCV(
            rf_base,
            param_grid,
            n_iter=10,  # Limit combinations for efficiency
            cv=3,  # 3-fold CV for speed
            scoring='average_precision',  # AUC-PR
            random_state=self.random_state,
            n_jobs=-1
        )
        
        rf_search.fit(self.X_train_scaled, self.y_train)
        
        self.rf_model = rf_search.best_estimator_
        
        print(f"Best parameters: {rf_search.best_params_}")
        print(f"Best CV AUC-PR: {rf_search.best_score_:.4f}")
        
        # Evaluate
        rf_results = self._evaluate_model(
            self.rf_model,
            self.X_test_scaled,
            self.y_test,
            "Random Forest"
        )
        
        self.models['Random Forest'] = self.rf_model
        self.results['Random Forest'] = rf_results
    
    def _train_xgboost(self):
        """Train XGBoost with hyperparameter tuning"""
        print("\n--- XGBOOST ---")
        print("Justification for XGBoost:")
        print("- State-of-the-art gradient boosting performance")
        print("- Built-in handling of missing values")
        print("- Feature importance and SHAP support")
        print("- Efficient implementation")
        print("- Excellent performance on structured data")
        
        # Calculate scale_pos_weight for class imbalance
        neg_count = (self.y_train == 0).sum()
        pos_count = (self.y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count
        
        print(f"\nClass imbalance handling:")
        print(f"Using scale_pos_weight = {scale_pos_weight:.2f}")
        print("This parameter balances positive and negative weights")
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
        
        print(f"\nPerforming hyperparameter tuning...")
        print(f"Parameter grid: {param_grid}")
        
        xgb_base = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            eval_metric='logloss',
            n_jobs=-1
        )
        
        # Use RandomizedSearchCV for efficiency
        xgb_search = RandomizedSearchCV(
            xgb_base,
            param_grid,
            n_iter=10,  # Limit combinations for efficiency
            cv=3,  # 3-fold CV for speed
            scoring='average_precision',  # AUC-PR
            random_state=self.random_state,
            n_jobs=-1
        )
        
        xgb_search.fit(self.X_train_scaled, self.y_train)
        
        self.xgb_model = xgb_search.best_estimator_
        
        print(f"Best parameters: {xgb_search.best_params_}")
        print(f"Best CV AUC-PR: {xgb_search.best_score_:.4f}")
        
        # Evaluate
        xgb_results = self._evaluate_model(
            self.xgb_model,
            self.X_test_scaled,
            self.y_test,
            "XGBoost"
        )
        
        self.models['XGBoost'] = self.xgb_model
        self.results['XGBoost'] = xgb_results
    
    def _evaluate_model(self, model, X_test, y_test, model_name):
        """Comprehensive model evaluation"""
        print(f"\n--- EVALUATING {model_name.upper()} ---")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        auc_pr = average_precision_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print(f"Performance Metrics:")
        print(f"AUC-PR (Primary): {auc_pr:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"True Negatives:  {tn:,}")
        print(f"False Positives: {fp:,}")
        print(f"False Negatives: {fn:,}")
        print(f"True Positives:  {tp:,}")
        
        # Business interpretation
        print(f"\nBusiness Interpretation:")
        print(f"- Correctly identified fraud cases: {tp:,} out of {tp+fn:,} ({recall*100:.1f}%)")
        print(f"- False alarms: {fp:,} out of {fp+tp:,} predictions ({fp/(fp+tp)*100:.1f}%)")
        print(f"- Missed fraud cases: {fn:,} (potential losses)")
        
        return {
            'auc_pr': auc_pr,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def perform_cross_validation(self):
        """
        Step 5: Cross-Validation for model stability assessment
        """
        print("\n" + "="*60)
        print("STEP 4: CROSS-VALIDATION")
        print("="*60)
        
        print("Performing Stratified K-Fold Cross-Validation (k=5)")
        print("Purpose: Assess model stability and generalization")
        print("Metrics: AUC-PR (primary) and F1-Score")
        
        # Setup stratified k-fold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        # Metrics to evaluate
        scoring = ['average_precision', 'f1']
        
        for model_name, model in self.models.items():
            print(f"\n--- {model_name.upper()} CROSS-VALIDATION ---")
            
            # Perform cross-validation
            cv_results = cross_validate(
                model, 
                self.X_train_scaled, 
                self.y_train,
                cv=skf,
                scoring=scoring,
                return_train_score=False,
                n_jobs=-1
            )
            
            # Calculate statistics
            auc_pr_scores = cv_results['test_average_precision']
            f1_scores = cv_results['test_f1']
            
            auc_pr_mean = auc_pr_scores.mean()
            auc_pr_std = auc_pr_scores.std()
            f1_mean = f1_scores.mean()
            f1_std = f1_scores.std()
            
            print(f"AUC-PR: {auc_pr_mean:.4f} ± {auc_pr_std:.4f}")
            print(f"F1-Score: {f1_mean:.4f} ± {f1_std:.4f}")
            
            # Store results
            self.cv_results[model_name] = {
                'auc_pr_mean': auc_pr_mean,
                'auc_pr_std': auc_pr_std,
                'f1_mean': f1_mean,
                'f1_std': f1_std,
                'auc_pr_scores': auc_pr_scores,
                'f1_scores': f1_scores
            }
            
            # Interpretation
            if auc_pr_std < 0.02:
                stability = "Very stable"
            elif auc_pr_std < 0.05:
                stability = "Stable"
            else:
                stability = "Variable (potential overfitting)"
            
            print(f"Stability assessment: {stability}")
    
    def compare_and_select_models(self):
        """
        Step 6: Model Comparison and Selection
        """
        print("\n" + "="*60)
        print("STEP 5: MODEL COMPARISON AND SELECTION")
        print("="*60)
        
        # Create comparison table
        comparison_data = []
        
        for model_name in self.models.keys():
            test_results = self.results[model_name]
            cv_results = self.cv_results[model_name]
            
            comparison_data.append({
                'Model': model_name,
                'Test_AUC_PR': test_results['auc_pr'],
                'Test_F1': test_results['f1_score'],
                'Test_Precision': test_results['precision'],
                'Test_Recall': test_results['recall'],
                'CV_AUC_PR_Mean': cv_results['auc_pr_mean'],
                'CV_AUC_PR_Std': cv_results['auc_pr_std'],
                'CV_F1_Mean': cv_results['f1_mean'],
                'CV_F1_Std': cv_results['f1_std']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\nMODEL COMPARISON TABLE:")
        print("="*100)
        print(comparison_df.round(4).to_string(index=False))
        
        # Model selection logic
        print(f"\n--- MODEL SELECTION CRITERIA ---")
        print("1. Highest AUC-PR (primary metric for imbalanced data)")
        print("2. Highest F1-Score (balance of precision and recall)")
        print("3. Lowest standard deviation in cross-validation (stability)")
        print("4. Business considerations (interpretability vs. performance)")
        
        # Find best model
        best_auc_pr_idx = comparison_df['Test_AUC_PR'].idxmax()
        best_model_name = comparison_df.loc[best_auc_pr_idx, 'Model']
        
        print(f"\n--- SELECTION DECISION ---")
        print(f"Best model by AUC-PR: {best_model_name}")
        
        # Detailed justification
        best_results = comparison_df.loc[best_auc_pr_idx]
        
        print(f"\nJustification for selecting {best_model_name}:")
        print(f"- AUC-PR: {best_results['Test_AUC_PR']:.4f} (highest)")
        print(f"- F1-Score: {best_results['Test_F1']:.4f}")
        print(f"- Cross-validation stability: ±{best_results['CV_AUC_PR_Std']:.4f}")
        
        # Performance improvement over baseline
        baseline_auc_pr = comparison_df[comparison_df['Model'] == 'Logistic Regression']['Test_AUC_PR'].iloc[0]
        improvement = ((best_results['Test_AUC_PR'] - baseline_auc_pr) / baseline_auc_pr) * 100
        
        print(f"- Improvement over baseline: {improvement:.1f}%")
        
        # Trade-off analysis
        if best_model_name == 'Logistic Regression':
            print("- High interpretability (coefficients)")
            print("- Fast prediction time")
            print("- Good for regulatory compliance")
        else:
            print("- Higher predictive performance")
            print("- More complex (ensemble method)")
            print("- Requires feature importance analysis for interpretability")
        
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        return comparison_df, best_model_name
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        print("\n" + "="*80)
        print("FINAL FRAUD DETECTION PIPELINE REPORT")
        print("="*80)
        
        print(f"\nDataset Summary:")
        print(f"- Total samples: {len(self.df):,}")
        print(f"- Features: {len(self.feature_names)}")
        print(f"- Class imbalance: {(self.df[self.target_col]==0).sum()/(self.df[self.target_col]==1).sum():.1f}:1")
        
        print(f"\nModels Evaluated:")
        for i, model_name in enumerate(self.models.keys(), 1):
            print(f"{i}. {model_name}")
        
        print(f"\nBest Model: {self.best_model_name}")
        best_results = self.results[self.best_model_name]
        print(f"- AUC-PR: {best_results['auc_pr']:.4f}")
        print(f"- F1-Score: {best_results['f1_score']:.4f}")
        print(f"- Precision: {best_results['precision']:.4f}")
        print(f"- Recall: {best_results['recall']:.4f}")
        
        print(f"\nBusiness Impact:")
        cm = best_results['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        print(f"- Fraud cases caught: {tp:,} out of {tp+fn:,} ({tp/(tp+fn)*100:.1f}%)")
        print(f"- False alarms: {fp:,} ({fp/(fp+tn)*100:.2f}% of non-fraud)")
        print(f"- Missed fraud: {fn:,} cases")
        
        print(f"\nRecommendations:")
        print("1. Deploy the selected model for real-time fraud detection")
        print("2. Monitor model performance and retrain periodically")
        print("3. Implement feedback loop for continuous improvement")
        print("4. Consider ensemble of top models for production")
        
        return {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'results': self.results,
            'cv_results': self.cv_results
        }
    
    def run_complete_pipeline(self, data_path):
        """
        Execute the complete fraud detection pipeline
        """
        try:
            # Step 1: Data Preparation
            self.load_and_explore_data(data_path)
            X, y = self.prepare_features()
            self.split_data(X, y)
            self.scale_features()
            
            # Step 2: Model Development
            self.train_baseline_model()
            self.train_ensemble_models()
            
            # Step 3: Cross-Validation
            self.perform_cross_validation()
            
            # Step 4: Model Comparison and Selection
            comparison_df, best_model_name = self.compare_and_select_models()
            
            # Step 5: Final Report
            final_report = self.generate_final_report()
            
            return final_report
            
        except Exception as e:
            print(f"Pipeline failed with error: {str(e)}")
            raise


def main():
    """
    Main execution function
    """
    # Initialize pipeline
    pipeline = FraudDetectionPipeline(random_state=42)
    
    # Check for available datasets
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
        print("Please ensure one of the following files exists:")
        for path in possible_paths:
            print(f"  - {path}")
        return
    
    print(f"Using dataset: {data_path}")
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline(data_path)
    
    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    return results


if __name__ == "__main__":
    results = main()