"""
Advanced Fraud Detection Pipeline with Visualization and Model Interpretation
===========================================================================

This enhanced version includes:
- Comprehensive visualizations (PR curves, feature importance)
- SHAP analysis for model interpretability
- Advanced ensemble methods
- Model persistence and loading
- Production-ready evaluation metrics

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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, roc_auc_score,
    roc_curve, plot_confusion_matrix
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
import os
import joblib
from datetime import datetime
import json

# Optional imports for advanced features
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")

warnings.filterwarnings('ignore')

class AdvancedFraudDetectionPipeline:
    """
    Advanced fraud detection pipeline with comprehensive analysis and visualization.
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
        os.makedirs('visualizations', exist_ok=True)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        print("="*80)
        print("ADVANCED FRAUD DETECTION PIPELINE INITIALIZED")
        print("="*80)
        print(f"Random State: {self.random_state}")
        print(f"SHAP Available: {SHAP_AVAILABLE}")
        print(f"LightGBM Available: {LIGHTGBM_AVAILABLE}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
    
    def load_and_explore_data(self, data_path):
        """Enhanced data loading with comprehensive EDA"""
        print("\n" + "="*60)
        print("STEP 1: ENHANCED DATA PREPARATION")
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
        
        # Enhanced EDA with visualizations
        self._perform_enhanced_eda()
        
        return self.df
    
    def _perform_enhanced_eda(self):
        """Comprehensive EDA with visualizations"""
        print("\n--- COMPREHENSIVE EXPLORATORY DATA ANALYSIS ---")
        
        # Basic info
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
        
        # Create visualizations
        self._create_eda_visualizations()
        
        # Missing values analysis
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print(f"\nMissing Values:")
            print(missing[missing > 0])
        else:
            print("\nNo missing values found")
        
        # Data types
        print(f"\nData Types:")
        print(self.df.dtypes.value_counts())
        
        # Correlation analysis for numerical features
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 1:
            print(f"\nNumerical Features: {len(numerical_cols)}")
            self._analyze_correlations(numerical_cols)
    
    def _create_eda_visualizations(self):
        """Create comprehensive EDA visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Class distribution
        class_counts = self.df[self.target_col].value_counts()
        axes[0, 0].pie(class_counts.values, labels=['Non-Fraud', 'Fraud'], autopct='%1.2f%%', startangle=90)
        axes[0, 0].set_title('Class Distribution')
        
        # Class distribution bar plot
        axes[0, 1].bar(['Non-Fraud', 'Fraud'], class_counts.values, color=['skyblue', 'salmon'])
        axes[0, 1].set_title('Class Distribution (Count)')
        axes[0, 1].set_ylabel('Count')
        
        # Feature distribution (if numerical features exist)
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 1:
            # Sample a few features for visualization
            sample_features = numerical_cols[:min(4, len(numerical_cols)-1)]  # Exclude target
            
            for i, feature in enumerate(sample_features):
                if i < 2:  # Only plot 2 features in remaining subplots
                    row, col = (1, i)
                    self.df.boxplot(column=feature, by=self.target_col, ax=axes[row, col])
                    axes[row, col].set_title(f'{feature} by Class')
                    axes[row, col].set_xlabel('Class')
        
        plt.tight_layout()
        plt.savefig('visualizations/eda_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("EDA visualizations saved to 'visualizations/eda_overview.png'")
    
    def _analyze_correlations(self, numerical_cols):
        """Analyze correlations between numerical features"""
        # Calculate correlation matrix
        corr_matrix = self.df[numerical_cols].corr()
        
        # Find highly correlated features
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > 0.8:  # High correlation threshold
                    high_corr_pairs.append((
                        corr_matrix.columns[i], 
                        corr_matrix.columns[j], 
                        corr_val
                    ))
        
        if high_corr_pairs:
            print(f"\nHighly Correlated Features (|r| > 0.8):")
            for feat1, feat2, corr_val in high_corr_pairs:
                print(f"  {feat1} - {feat2}: {corr_val:.3f}")
        
        # Create correlation heatmap for top features
        if len(numerical_cols) > 10:
            # Select top features by correlation with target
            target_corr = abs(corr_matrix[self.target_col]).sort_values(ascending=False)
            top_features = target_corr.head(10).index
            corr_subset = self.df[top_features].corr()
        else:
            corr_subset = corr_matrix
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_subset, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('visualizations/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def prepare_features(self):
        """Enhanced feature preparation"""
        print("\n--- ENHANCED FEATURE PREPARATION ---")
        
        # Separate features and target
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        
        # Handle missing values
        if X.isnull().sum().sum() > 0:
            print("Handling missing values...")
            numerical_cols = X.select_dtypes(include=[np.number]).columns
            categorical_cols = X.select_dtypes(include=['object']).columns
            
            # Numerical: median imputation
            X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].median())
            
            # Categorical: mode or 'Unknown'
            for col in categorical_cols:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown')
        
        # Feature engineering for specific datasets
        if 'Amount' in X.columns:  # creditcard.csv
            print("Detected creditcard.csv format - applying specific preprocessing")
            # Log transform for Amount (common in fraud detection)
            X['Amount_log'] = np.log1p(X['Amount'])
            
        # Store feature information
        self.feature_names = X.columns.tolist()
        self.numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        print(f"Features prepared: {len(self.feature_names)}")
        print(f"Numerical features: {len(self.numerical_features)}")
        print(f"Categorical features: {len(self.categorical_features)}")
        
        return X, y
    
    def split_data(self, X, y):
        """Enhanced data splitting with detailed analysis"""
        print("\n--- ENHANCED DATA SPLITTING ---")
        
        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=self.random_state,
            stratify=y,
            shuffle=True
        )
        
        # Detailed split analysis
        print(f"Training set: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
        print(f"Test set: {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
        
        # Class distribution analysis
        train_dist = y_train.value_counts(normalize=True)
        test_dist = y_test.value_counts(normalize=True)
        
        print(f"\nClass Distribution Comparison:")
        print(f"                 Training    Test     Difference")
        print(f"Non-fraud:       {train_dist[0]:.3f}     {test_dist[0]:.3f}    {abs(train_dist[0]-test_dist[0]):.3f}")
        print(f"Fraud:           {train_dist[1]:.3f}     {test_dist[1]:.3f}    {abs(train_dist[1]-test_dist[1]):.3f}")
        
        # Validate stratification quality
        max_diff = max(abs(train_dist[0]-test_dist[0]), abs(train_dist[1]-test_dist[1]))
        if max_diff < 0.01:
            print("✓ Excellent stratification (difference < 1%)")
        elif max_diff < 0.02:
            print("✓ Good stratification (difference < 2%)")
        else:
            print("⚠ Stratification could be improved")
        
        # Store splits
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self):
        """Enhanced feature scaling with analysis"""
        print("\n--- ENHANCED FEATURE SCALING ---")
        
        if len(self.numerical_features) > 0:
            self.scaler = StandardScaler()
            
            # Create scaled copies
            self.X_train_scaled = self.X_train.copy()
            self.X_test_scaled = self.X_test.copy()
            
            # Fit and transform
            self.X_train_scaled[self.numerical_features] = self.scaler.fit_transform(
                self.X_train[self.numerical_features]
            )
            self.X_test_scaled[self.numerical_features] = self.scaler.transform(
                self.X_test[self.numerical_features]
            )
            
            print(f"Scaled {len(self.numerical_features)} numerical features")
            
            # Show scaling statistics
            print("\nScaling Statistics (mean ± std after scaling):")
            scaled_stats = self.X_train_scaled[self.numerical_features].describe()
            print(f"Mean range: [{scaled_stats.loc['mean'].min():.3f}, {scaled_stats.loc['mean'].max():.3f}]")
            print(f"Std range: [{scaled_stats.loc['std'].min():.3f}, {scaled_stats.loc['std'].max():.3f}]")
            
        else:
            self.X_train_scaled = self.X_train
            self.X_test_scaled = self.X_test
            print("No numerical features to scale")
    
    def train_baseline_model(self):
        """Enhanced baseline model with detailed analysis"""
        print("\n" + "="*60)
        print("STEP 2: ENHANCED BASELINE MODEL")
        print("="*60)
        
        print("\n--- LOGISTIC REGRESSION BASELINE ---")
        
        # Calculate class weights
        class_counts = self.y_train.value_counts()
        total_samples = len(self.y_train)
        weight_0 = total_samples / (2 * class_counts[0])
        weight_1 = total_samples / (2 * class_counts[1])
        
        print(f"Class weight calculation:")
        print(f"Class 0 weight: {weight_0:.3f}")
        print(f"Class 1 weight: {weight_1:.3f}")
        print(f"This gives {weight_1/weight_0:.1f}x more importance to fraud cases")
        
        # Train model
        self.baseline_model = LogisticRegression(
            class_weight='balanced',
            random_state=self.random_state,
            max_iter=1000,
            solver='liblinear'  # Good for small datasets
        )
        
        self.baseline_model.fit(self.X_train_scaled, self.y_train)
        
        # Feature importance analysis
        if len(self.numerical_features) <= 20:  # Only for manageable number of features
            self._analyze_logistic_regression_coefficients()
        
        # Evaluate
        baseline_results = self._evaluate_model_enhanced(
            self.baseline_model, 
            self.X_test_scaled, 
            self.y_test, 
            "Logistic Regression"
        )
        
        self.models['Logistic Regression'] = self.baseline_model
        self.results['Logistic Regression'] = baseline_results
        
        return self.baseline_model
    
    def _analyze_logistic_regression_coefficients(self):
        """Analyze logistic regression coefficients for interpretability"""
        print("\n--- FEATURE IMPORTANCE ANALYSIS ---")
        
        # Get coefficients
        coefficients = self.baseline_model.coef_[0]
        feature_importance = pd.DataFrame({
            'feature': self.X_train_scaled.columns,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False)
        
        print("Top 10 Most Important Features (by |coefficient|):")
        print(feature_importance.head(10)[['feature', 'coefficient']].to_string(index=False))
        
        # Visualize top features
        top_features = feature_importance.head(10)
        
        plt.figure(figsize=(10, 6))
        colors = ['red' if x < 0 else 'blue' for x in top_features['coefficient']]
        plt.barh(range(len(top_features)), top_features['coefficient'], color=colors)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Coefficient Value')
        plt.title('Top 10 Feature Coefficients (Logistic Regression)')
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('visualizations/logistic_regression_coefficients.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def train_ensemble_models(self):
        """Enhanced ensemble model training"""
        print("\n" + "="*60)
        print("STEP 3: ENHANCED ENSEMBLE MODELS")
        print("="*60)
        
        # Random Forest
        self._train_random_forest_enhanced()
        
        # XGBoost
        self._train_xgboost_enhanced()
        
        # LightGBM (if available)
        if LIGHTGBM_AVAILABLE:
            self._train_lightgbm()
        
        # Ensemble of models
        self._create_voting_ensemble()
    
    def _train_random_forest_enhanced(self):
        """Enhanced Random Forest training"""
        print("\n--- ENHANCED RANDOM FOREST ---")
        
        # Calculate class weights
        neg_count = (self.y_train == 0).sum()
        pos_count = (self.y_train == 1).sum()
        
        print(f"Class imbalance handling:")
        print(f"Negative samples: {neg_count:,}")
        print(f"Positive samples: {pos_count:,}")
        print(f"Using class_weight='balanced'")
        
        # Enhanced hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        print(f"Hyperparameter tuning with {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * len(param_grid['max_features'])} combinations")
        
        rf_base = RandomForestClassifier(
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Use RandomizedSearchCV for efficiency
        rf_search = RandomizedSearchCV(
            rf_base,
            param_grid,
            n_iter=20,  # More iterations for better tuning
            cv=5,
            scoring='average_precision',
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1
        )
        
        rf_search.fit(self.X_train_scaled, self.y_train)
        
        self.rf_model = rf_search.best_estimator_
        
        print(f"Best parameters: {rf_search.best_params_}")
        print(f"Best CV AUC-PR: {rf_search.best_score_:.4f}")
        
        # Feature importance analysis
        self._analyze_tree_feature_importance(self.rf_model, "Random Forest")
        
        # Evaluate
        rf_results = self._evaluate_model_enhanced(
            self.rf_model,
            self.X_test_scaled,
            self.y_test,
            "Random Forest"
        )
        
        self.models['Random Forest'] = self.rf_model
        self.results['Random Forest'] = rf_results
    
    def _train_xgboost_enhanced(self):
        """Enhanced XGBoost training"""
        print("\n--- ENHANCED XGBOOST ---")
        
        # Calculate scale_pos_weight
        neg_count = (self.y_train == 0).sum()
        pos_count = (self.y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count
        
        print(f"Scale pos weight: {scale_pos_weight:.2f}")
        
        # Enhanced hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        xgb_base = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            eval_metric='logloss',
            n_jobs=-1
        )
        
        xgb_search = RandomizedSearchCV(
            xgb_base,
            param_grid,
            n_iter=20,
            cv=5,
            scoring='average_precision',
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1
        )
        
        xgb_search.fit(self.X_train_scaled, self.y_train)
        
        self.xgb_model = xgb_search.best_estimator_
        
        print(f"Best parameters: {xgb_search.best_params_}")
        print(f"Best CV AUC-PR: {xgb_search.best_score_:.4f}")
        
        # Feature importance analysis
        self._analyze_tree_feature_importance(self.xgb_model, "XGBoost")
        
        # Evaluate
        xgb_results = self._evaluate_model_enhanced(
            self.xgb_model,
            self.X_test_scaled,
            self.y_test,
            "XGBoost"
        )
        
        self.models['XGBoost'] = self.xgb_model
        self.results['XGBoost'] = xgb_results
    
    def _train_lightgbm(self):
        """Train LightGBM model"""
        print("\n--- LIGHTGBM ---")
        
        # Calculate class weights
        neg_count = (self.y_train == 0).sum()
        pos_count = (self.y_train == 1).sum()
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10],
            'learning_rate': [0.01, 0.1],
            'num_leaves': [31, 50]
        }
        
        lgb_base = lgb.LGBMClassifier(
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1
        )
        
        lgb_search = RandomizedSearchCV(
            lgb_base,
            param_grid,
            n_iter=10,
            cv=3,
            scoring='average_precision',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        lgb_search.fit(self.X_train_scaled, self.y_train)
        
        self.lgb_model = lgb_search.best_estimator_
        
        print(f"Best parameters: {lgb_search.best_params_}")
        print(f"Best CV AUC-PR: {lgb_search.best_score_:.4f}")
        
        # Evaluate
        lgb_results = self._evaluate_model_enhanced(
            self.lgb_model,
            self.X_test_scaled,
            self.y_test,
            "LightGBM"
        )
        
        self.models['LightGBM'] = self.lgb_model
        self.results['LightGBM'] = lgb_results
    
    def _create_voting_ensemble(self):
        """Create voting ensemble of best models"""
        print("\n--- VOTING ENSEMBLE ---")
        
        # Select top 3 models for ensemble
        model_scores = [(name, results['auc_pr']) for name, results in self.results.items()]
        model_scores.sort(key=lambda x: x[1], reverse=True)
        top_models = model_scores[:3]
        
        print(f"Creating ensemble from top 3 models:")
        for name, score in top_models:
            print(f"  {name}: {score:.4f}")
        
        # Create voting classifier
        estimators = [(name, self.models[name]) for name, _ in top_models]
        
        self.voting_model = VotingClassifier(
            estimators=estimators,
            voting='soft',  # Use probability predictions
            n_jobs=-1
        )
        
        self.voting_model.fit(self.X_train_scaled, self.y_train)
        
        # Evaluate
        voting_results = self._evaluate_model_enhanced(
            self.voting_model,
            self.X_test_scaled,
            self.y_test,
            "Voting Ensemble"
        )
        
        self.models['Voting Ensemble'] = self.voting_model
        self.results['Voting Ensemble'] = voting_results
    
    def _analyze_tree_feature_importance(self, model, model_name):
        """Analyze feature importance for tree-based models"""
        print(f"\n--- {model_name.upper()} FEATURE IMPORTANCE ---")
        
        # Get feature importance
        importance_scores = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.X_train_scaled.columns,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        print("Top 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        # Visualize top features
        top_features = feature_importance.head(10)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top 10 Feature Importance ({model_name})')
        plt.tight_layout()
        plt.savefig(f'visualizations/{model_name.lower().replace(" ", "_")}_feature_importance.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def _evaluate_model_enhanced(self, model, X_test, y_test, model_name):
        """Enhanced model evaluation with visualizations"""
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
        
        # Business metrics
        total_fraud = tp + fn
        total_legit = tn + fp
        fraud_detection_rate = tp / total_fraud if total_fraud > 0 else 0
        false_alarm_rate = fp / total_legit if total_legit > 0 else 0
        
        print(f"\nBusiness Metrics:")
        print(f"Fraud Detection Rate: {fraud_detection_rate*100:.1f}%")
        print(f"False Alarm Rate: {false_alarm_rate*100:.2f}%")
        
        # Create evaluation visualizations
        self._create_evaluation_plots(y_test, y_pred, y_pred_proba, model_name)
        
        return {
            'auc_pr': auc_pr,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'fraud_detection_rate': fraud_detection_rate,
            'false_alarm_rate': false_alarm_rate
        }
    
    def _create_evaluation_plots(self, y_test, y_pred, y_pred_proba, model_name):
        """Create comprehensive evaluation plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title(f'Confusion Matrix - {model_name}')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        axes[0, 1].plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title(f'ROC Curve - {model_name}')
        axes[0, 1].legend()
        
        # Precision-Recall Curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
        auc_pr = average_precision_score(y_test, y_pred_proba)
        axes[1, 0].plot(recall_curve, precision_curve, label=f'PR Curve (AUC = {auc_pr:.3f})')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title(f'Precision-Recall Curve - {model_name}')
        axes[1, 0].legend()
        
        # Prediction Distribution
        axes[1, 1].hist(y_pred_proba[y_test == 0], bins=50, alpha=0.7, label='Non-Fraud', density=True)
        axes[1, 1].hist(y_pred_proba[y_test == 1], bins=50, alpha=0.7, label='Fraud', density=True)
        axes[1, 1].set_xlabel('Prediction Probability')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title(f'Prediction Distribution - {model_name}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(f'visualizations/{model_name.lower().replace(" ", "_")}_evaluation.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def perform_cross_validation(self):
        """Enhanced cross-validation with detailed analysis"""
        print("\n" + "="*60)
        print("STEP 4: ENHANCED CROSS-VALIDATION")
        print("="*60)
        
        print("Performing Stratified K-Fold Cross-Validation (k=5)")
        print("Metrics: AUC-PR, F1-Score, Precision, Recall")
        
        # Setup stratified k-fold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        # Enhanced scoring metrics
        scoring = ['average_precision', 'f1', 'precision', 'recall', 'roc_auc']
        
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
            
            # Calculate statistics for each metric
            results_summary = {}
            for metric in scoring:
                scores = cv_results[f'test_{metric}']
                results_summary[metric] = {
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'min': scores.min(),
                    'max': scores.max(),
                    'scores': scores
                }
            
            # Display results
            print(f"Cross-Validation Results (5-fold):")
            for metric, stats in results_summary.items():
                print(f"{metric:>15}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                     f"(min: {stats['min']:.4f}, max: {stats['max']:.4f})")
            
            # Store results
            self.cv_results[model_name] = results_summary
            
            # Stability assessment
            auc_pr_std = results_summary['average_precision']['std']
            if auc_pr_std < 0.01:
                stability = "Excellent stability"
            elif auc_pr_std < 0.02:
                stability = "Good stability"
            elif auc_pr_std < 0.05:
                stability = "Moderate stability"
            else:
                stability = "Poor stability (potential overfitting)"
            
            print(f"Stability Assessment: {stability}")
    
    def compare_and_select_models(self):
        """Enhanced model comparison with comprehensive analysis"""
        print("\n" + "="*60)
        print("STEP 5: ENHANCED MODEL COMPARISON")
        print("="*60)
        
        # Create comprehensive comparison table
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
                'Test_ROC_AUC': test_results['roc_auc'],
                'CV_AUC_PR_Mean': cv_results['average_precision']['mean'],
                'CV_AUC_PR_Std': cv_results['average_precision']['std'],
                'CV_F1_Mean': cv_results['f1']['mean'],
                'CV_F1_Std': cv_results['f1']['std'],
                'Fraud_Detection_Rate': test_results['fraud_detection_rate'],
                'False_Alarm_Rate': test_results['false_alarm_rate']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\nCOMPREHENSIVE MODEL COMPARISON TABLE:")
        print("="*120)
        print(comparison_df.round(4).to_string(index=False))
        
        # Save comparison table
        comparison_df.to_csv('models/model_comparison.csv', index=False)
        print(f"\nComparison table saved to 'models/model_comparison.csv'")
        
        # Advanced model selection
        self._perform_advanced_model_selection(comparison_df)
        
        # Create comparison visualizations
        self._create_comparison_visualizations(comparison_df)
        
        return comparison_df, self.best_model_name
    
    def _perform_advanced_model_selection(self, comparison_df):
        """Advanced model selection with multiple criteria"""
        print(f"\n--- ADVANCED MODEL SELECTION ---")
        
        # Multi-criteria scoring
        weights = {
            'auc_pr': 0.4,      # Primary metric
            'f1_score': 0.2,    # Balance metric
            'stability': 0.2,   # Consistency
            'recall': 0.2       # Fraud detection capability
        }
        
        print(f"Selection Criteria Weights:")
        for criterion, weight in weights.items():
            print(f"  {criterion}: {weight*100:.0f}%")
        
        # Calculate composite scores
        scores = []
        for _, row in comparison_df.iterrows():
            # Normalize metrics to 0-1 scale
            auc_pr_norm = row['Test_AUC_PR']
            f1_norm = row['Test_F1']
            stability_norm = 1 - min(row['CV_AUC_PR_Std'] / 0.1, 1)  # Lower std = higher score
            recall_norm = row['Test_Recall']
            
            composite_score = (
                weights['auc_pr'] * auc_pr_norm +
                weights['f1_score'] * f1_norm +
                weights['stability'] * stability_norm +
                weights['recall'] * recall_norm
            )
            
            scores.append(composite_score)
        
        comparison_df['Composite_Score'] = scores
        
        # Select best model
        best_idx = comparison_df['Composite_Score'].idxmax()
        self.best_model_name = comparison_df.loc[best_idx, 'Model']
        self.best_model = self.models[self.best_model_name]
        
        print(f"\nModel Rankings by Composite Score:")
        ranking = comparison_df.sort_values('Composite_Score', ascending=False)
        for i, (_, row) in enumerate(ranking.iterrows(), 1):
            print(f"{i}. {row['Model']}: {row['Composite_Score']:.4f}")
        
        print(f"\nSelected Best Model: {self.best_model_name}")
        
        # Detailed justification
        best_row = comparison_df.loc[best_idx]
        print(f"\nDetailed Justification:")
        print(f"- AUC-PR: {best_row['Test_AUC_PR']:.4f} (weight: {weights['auc_pr']*100:.0f}%)")
        print(f"- F1-Score: {best_row['Test_F1']:.4f} (weight: {weights['f1_score']*100:.0f}%)")
        print(f"- Stability: ±{best_row['CV_AUC_PR_Std']:.4f} (weight: {weights['stability']*100:.0f}%)")
        print(f"- Recall: {best_row['Test_Recall']:.4f} (weight: {weights['recall']*100:.0f}%)")
        print(f"- Composite Score: {best_row['Composite_Score']:.4f}")
    
    def _create_comparison_visualizations(self, comparison_df):
        """Create comprehensive comparison visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # AUC-PR Comparison
        axes[0, 0].bar(comparison_df['Model'], comparison_df['Test_AUC_PR'])
        axes[0, 0].set_title('AUC-PR Comparison')
        axes[0, 0].set_ylabel('AUC-PR')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # F1-Score Comparison
        axes[0, 1].bar(comparison_df['Model'], comparison_df['Test_F1'])
        axes[0, 1].set_title('F1-Score Comparison')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Stability Comparison (CV Standard Deviation)
        axes[1, 0].bar(comparison_df['Model'], comparison_df['CV_AUC_PR_Std'])
        axes[1, 0].set_title('Model Stability (Lower is Better)')
        axes[1, 0].set_ylabel('CV AUC-PR Std Dev')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Composite Score
        axes[1, 1].bar(comparison_df['Model'], comparison_df['Composite_Score'])
        axes[1, 1].set_title('Composite Score (Higher is Better)')
        axes[1, 1].set_ylabel('Composite Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def perform_shap_analysis(self):
        """Perform SHAP analysis for model interpretability"""
        if not SHAP_AVAILABLE:
            print("SHAP not available. Skipping interpretability analysis.")
            return
        
        print("\n" + "="*60)
        print("STEP 6: MODEL INTERPRETABILITY (SHAP)")
        print("="*60)
        
        print(f"Performing SHAP analysis for {self.best_model_name}")
        
        # Create SHAP explainer
        if hasattr(self.best_model, 'predict_proba'):
            explainer = shap.Explainer(self.best_model, self.X_train_scaled.sample(100))
        else:
            explainer = shap.LinearExplainer(self.best_model, self.X_train_scaled)
        
        # Calculate SHAP values for test set sample
        test_sample = self.X_test_scaled.sample(min(500, len(self.X_test_scaled)))
        shap_values = explainer(test_sample)
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, test_sample, show=False)
        plt.tight_layout()
        plt.savefig('visualizations/shap_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Feature importance plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, test_sample, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig('visualizations/shap_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("SHAP analysis completed. Plots saved to visualizations/")
    
    def save_models_and_results(self):
        """Save models and results for production use"""
        print("\n--- SAVING MODELS AND RESULTS ---")
        
        # Save best model
        joblib.dump(self.best_model, f'models/best_model_{self.best_model_name.lower().replace(" ", "_")}.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        
        # Save all models
        for name, model in self.models.items():
            filename = f'models/{name.lower().replace(" ", "_")}.pkl'
            joblib.dump(model, filename)
        
        # Save results
        with open('models/results.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            results_serializable = {}
            for model_name, results in self.results.items():
                results_copy = results.copy()
                results_copy['confusion_matrix'] = results_copy['confusion_matrix'].tolist()
                results_copy['y_pred'] = results_copy['y_pred'].tolist()
                results_copy['y_pred_proba'] = results_copy['y_pred_proba'].tolist()
                results_serializable[model_name] = results_copy
            
            json.dump(results_serializable, f, indent=2)
        
        # Save metadata
        metadata = {
            'best_model': self.best_model_name,
            'feature_names': self.feature_names,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'target_column': self.target_col,
            'random_state': self.random_state,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('models/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("Models and results saved to 'models/' directory")
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE FRAUD DETECTION PIPELINE REPORT")
        print("="*80)
        
        print(f"\nExecution Summary:")
        print(f"- Pipeline completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"- Dataset: {self.df.shape[0]:,} samples, {len(self.feature_names)} features")
        print(f"- Class imbalance: {(self.df[self.target_col]==0).sum()/(self.df[self.target_col]==1).sum():.1f}:1")
        
        print(f"\nModels Evaluated: {len(self.models)}")
        for i, model_name in enumerate(self.models.keys(), 1):
            results = self.results[model_name]
            print(f"{i}. {model_name}: AUC-PR={results['auc_pr']:.4f}, F1={results['f1_score']:.4f}")
        
        print(f"\nBest Model: {self.best_model_name}")
        best_results = self.results[self.best_model_name]
        print(f"Performance Metrics:")
        print(f"- AUC-PR: {best_results['auc_pr']:.4f}")
        print(f"- F1-Score: {best_results['f1_score']:.4f}")
        print(f"- Precision: {best_results['precision']:.4f}")
        print(f"- Recall: {best_results['recall']:.4f}")
        print(f"- ROC-AUC: {best_results['roc_auc']:.4f}")
        
        print(f"\nBusiness Impact:")
        print(f"- Fraud Detection Rate: {best_results['fraud_detection_rate']*100:.1f}%")
        print(f"- False Alarm Rate: {best_results['false_alarm_rate']*100:.2f}%")
        
        cm = best_results['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        print(f"- True Positives (Caught Fraud): {tp:,}")
        print(f"- False Negatives (Missed Fraud): {fn:,}")
        print(f"- False Positives (False Alarms): {fp:,}")
        
        print(f"\nCross-Validation Stability:")
        cv_results = self.cv_results[self.best_model_name]
        print(f"- AUC-PR: {cv_results['average_precision']['mean']:.4f} ± {cv_results['average_precision']['std']:.4f}")
        print(f"- F1-Score: {cv_results['f1']['mean']:.4f} ± {cv_results['f1']['std']:.4f}")
        
        print(f"\nFiles Generated:")
        print("- models/: Trained models and preprocessing objects")
        print("- visualizations/: Comprehensive analysis plots")
        print("- models/model_comparison.csv: Detailed comparison table")
        print("- models/results.json: Complete results data")
        
        print(f"\nProduction Recommendations:")
        print("1. Deploy the selected model with real-time monitoring")
        print("2. Implement A/B testing against current system")
        print("3. Set up automated retraining pipeline")
        print("4. Monitor for data drift and model degradation")
        print("5. Establish feedback loop for continuous improvement")
        
        if best_results['recall'] > 0.9:
            print("6. ✓ High recall achieved - good fraud detection capability")
        else:
            print("6. ⚠ Consider adjusting threshold to improve recall")
        
        if best_results['false_alarm_rate'] < 0.05:
            print("7. ✓ Low false alarm rate - minimal customer friction")
        else:
            print("7. ⚠ Consider balancing precision vs recall based on business needs")
        
        return {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'results': self.results,
            'cv_results': self.cv_results,
            'feature_names': self.feature_names
        }
    
    def run_complete_pipeline(self, data_path):
        """Execute the complete advanced fraud detection pipeline"""
        try:
            # Step 1: Enhanced Data Preparation
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
            
            # Step 5: Model Interpretability
            self.perform_shap_analysis()
            
            # Step 6: Save Everything
            self.save_models_and_results()
            
            # Step 7: Final Report
            final_report = self.generate_final_report()
            
            return final_report
            
        except Exception as e:
            print(f"Pipeline failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """Main execution function for advanced pipeline"""
    # Initialize advanced pipeline
    pipeline = AdvancedFraudDetectionPipeline(random_state=42)
    
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
    
    # Run complete advanced pipeline
    results = pipeline.run_complete_pipeline(data_path)
    
    print("\n" + "="*80)
    print("ADVANCED PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    return results


if __name__ == "__main__":
    results = main()