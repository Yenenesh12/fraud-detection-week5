"""
Unit tests for fraud detection pipeline.

This module contains comprehensive tests for the fraud detection system
to ensure reliability and correctness.

Author: AI Assistant
Date: 2026-01-02
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from fraud_detection_pipeline import FraudDetectionPipeline
from fraud_detection_advanced import AdvancedFraudDetectionPipeline


class TestFraudDetectionPipeline(unittest.TestCase):
    """Test cases for the basic fraud detection pipeline."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests."""
        # Create synthetic test data
        np.random.seed(42)
        n_samples = 1000
        n_fraud = 30  # 3% fraud rate
        n_normal = n_samples - n_fraud
        
        data = []
        
        # Normal transactions
        for i in range(n_normal):
            data.append({
                'Amount': max(np.random.lognormal(3, 1), 1),
                'V1': np.random.normal(0, 1),
                'V2': np.random.normal(0, 1),
                'V3': np.random.normal(0, 1),
                'V4': np.random.normal(0, 1),
                'Class': 0
            })
        
        # Fraud transactions
        for i in range(n_fraud):
            data.append({
                'Amount': max(np.random.lognormal(5, 1.5), 1),
                'V1': np.random.normal(2, 1),
                'V2': np.random.normal(-1, 1),
                'V3': np.random.normal(1, 1),
                'V4': np.random.normal(-2, 1),
                'Class': 1
            })
        
        cls.test_df = pd.DataFrame(data).sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save test data
        os.makedirs('test_data', exist_ok=True)
        cls.test_data_path = 'test_data/test_fraud_data.csv'
        cls.test_df.to_csv(cls.test_data_path, index=False)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        # Remove test data
        if os.path.exists(cls.test_data_path):
            os.remove(cls.test_data_path)
        if os.path.exists('test_data'):
            os.rmdir('test_data')
    
    def setUp(self):
        """Set up for each test."""
        self.pipeline = FraudDetectionPipeline(random_state=42)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        self.assertEqual(self.pipeline.random_state, 42)
        self.assertIsInstance(self.pipeline.models, dict)
        self.assertIsInstance(self.pipeline.results, dict)
        self.assertIsInstance(self.pipeline.cv_results, dict)
    
    def test_data_loading(self):
        """Test data loading functionality."""
        df = self.pipeline.load_and_explore_data(self.test_data_path)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1000)
        self.assertIn('Class', df.columns)
        self.assertEqual(self.pipeline.target_col, 'Class')
    
    def test_feature_preparation(self):
        """Test feature preparation."""
        self.pipeline.load_and_explore_data(self.test_data_path)
        X, y = self.pipeline.prepare_features()
        
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(len(X), len(y))
        self.assertNotIn('Class', X.columns)
        self.assertEqual(len(self.pipeline.feature_names), X.shape[1])
    
    def test_data_splitting(self):
        """Test data splitting functionality."""
        self.pipeline.load_and_explore_data(self.test_data_path)
        X, y = self.pipeline.prepare_features()
        X_train, X_test, y_train, y_test = self.pipeline.split_data(X, y)
        
        # Check split sizes
        self.assertEqual(len(X_train), 800)  # 80% of 1000
        self.assertEqual(len(X_test), 200)   # 20% of 1000
        self.assertEqual(len(y_train), 800)
        self.assertEqual(len(y_test), 200)
        
        # Check stratification (approximate due to small sample)
        train_fraud_rate = y_train.mean()
        test_fraud_rate = y_test.mean()
        overall_fraud_rate = y.mean()
        
        self.assertAlmostEqual(train_fraud_rate, overall_fraud_rate, delta=0.02)
        self.assertAlmostEqual(test_fraud_rate, overall_fraud_rate, delta=0.02)
    
    def test_feature_scaling(self):
        """Test feature scaling."""
        self.pipeline.load_and_explore_data(self.test_data_path)
        X, y = self.pipeline.prepare_features()
        self.pipeline.split_data(X, y)
        self.pipeline.scale_features()
        
        # Check that scaler exists
        self.assertTrue(hasattr(self.pipeline, 'scaler'))
        
        # Check that scaled data exists
        self.assertTrue(hasattr(self.pipeline, 'X_train_scaled'))
        self.assertTrue(hasattr(self.pipeline, 'X_test_scaled'))
        
        # Check scaling (mean should be close to 0, std close to 1)
        numerical_cols = self.pipeline.X_train.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            scaled_means = self.pipeline.X_train_scaled[numerical_cols].mean()
            scaled_stds = self.pipeline.X_train_scaled[numerical_cols].std()
            
            for mean_val in scaled_means:
                self.assertAlmostEqual(mean_val, 0, delta=0.1)
            for std_val in scaled_stds:
                self.assertAlmostEqual(std_val, 1, delta=0.1)
    
    def test_baseline_model_training(self):
        """Test baseline model training."""
        self.pipeline.load_and_explore_data(self.test_data_path)
        X, y = self.pipeline.prepare_features()
        self.pipeline.split_data(X, y)
        self.pipeline.scale_features()
        
        model = self.pipeline.train_baseline_model()
        
        # Check model exists
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'predict'))
        self.assertTrue(hasattr(model, 'predict_proba'))
        
        # Check results stored
        self.assertIn('Logistic Regression', self.pipeline.models)
        self.assertIn('Logistic Regression', self.pipeline.results)
        
        # Check result structure
        results = self.pipeline.results['Logistic Regression']
        required_keys = ['auc_pr', 'f1_score', 'precision', 'recall', 'roc_auc']
        for key in required_keys:
            self.assertIn(key, results)
            self.assertIsInstance(results[key], (int, float))
            self.assertGreaterEqual(results[key], 0)
            self.assertLessEqual(results[key], 1)
    
    def test_model_evaluation_metrics(self):
        """Test model evaluation metrics."""
        self.pipeline.load_and_explore_data(self.test_data_path)
        X, y = self.pipeline.prepare_features()
        self.pipeline.split_data(X, y)
        self.pipeline.scale_features()
        self.pipeline.train_baseline_model()
        
        results = self.pipeline.results['Logistic Regression']
        
        # Check metric ranges
        self.assertGreaterEqual(results['auc_pr'], 0)
        self.assertLessEqual(results['auc_pr'], 1)
        self.assertGreaterEqual(results['f1_score'], 0)
        self.assertLessEqual(results['f1_score'], 1)
        self.assertGreaterEqual(results['precision'], 0)
        self.assertLessEqual(results['precision'], 1)
        self.assertGreaterEqual(results['recall'], 0)
        self.assertLessEqual(results['recall'], 1)
        
        # Check confusion matrix
        cm = results['confusion_matrix']
        self.assertEqual(cm.shape, (2, 2))
        self.assertEqual(cm.sum(), len(self.pipeline.y_test))
    
    def test_cross_validation(self):
        """Test cross-validation functionality."""
        self.pipeline.load_and_explore_data(self.test_data_path)
        X, y = self.pipeline.prepare_features()
        self.pipeline.split_data(X, y)
        self.pipeline.scale_features()
        self.pipeline.train_baseline_model()
        self.pipeline.perform_cross_validation()
        
        # Check CV results exist
        self.assertIn('Logistic Regression', self.pipeline.cv_results)
        
        cv_results = self.pipeline.cv_results['Logistic Regression']
        required_keys = ['auc_pr_mean', 'auc_pr_std', 'f1_mean', 'f1_std']
        
        for key in required_keys:
            self.assertIn(key, cv_results)
            self.assertIsInstance(cv_results[key], (int, float))
    
    def test_complete_pipeline(self):
        """Test complete pipeline execution."""
        results = self.pipeline.run_complete_pipeline(self.test_data_path)
        
        # Check results structure
        self.assertIn('best_model', results)
        self.assertIn('best_model_name', results)
        self.assertIn('results', results)
        self.assertIn('cv_results', results)
        
        # Check that best model exists
        self.assertIsNotNone(results['best_model'])
        self.assertIsInstance(results['best_model_name'], str)
        
        # Check that multiple models were trained
        self.assertGreaterEqual(len(results['results']), 2)  # At least baseline + 1 ensemble


class TestAdvancedFraudDetectionPipeline(unittest.TestCase):
    """Test cases for the advanced fraud detection pipeline."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Use same test data as basic pipeline
        TestFraudDetectionPipeline.setUpClass()
        cls.test_df = TestFraudDetectionPipeline.test_df
        cls.test_data_path = TestFraudDetectionPipeline.test_data_path
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        TestFraudDetectionPipeline.tearDownClass()
    
    def setUp(self):
        """Set up for each test."""
        self.pipeline = AdvancedFraudDetectionPipeline(random_state=42)
    
    def test_advanced_pipeline_initialization(self):
        """Test advanced pipeline initialization."""
        self.assertEqual(self.pipeline.random_state, 42)
        self.assertIsInstance(self.pipeline.models, dict)
        self.assertIsInstance(self.pipeline.results, dict)
        self.assertIsInstance(self.pipeline.cv_results, dict)
    
    def test_enhanced_eda(self):
        """Test enhanced EDA functionality."""
        df = self.pipeline.load_and_explore_data(self.test_data_path)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(self.pipeline.target_col, 'Class')
        
        # Check that feature information is stored
        X, y = self.pipeline.prepare_features()
        self.assertTrue(hasattr(self.pipeline, 'feature_names'))
        self.assertTrue(hasattr(self.pipeline, 'numerical_features'))
        self.assertTrue(hasattr(self.pipeline, 'categorical_features'))
    
    def test_model_comparison_and_selection(self):
        """Test model comparison and selection."""
        # Run a minimal pipeline to test comparison
        self.pipeline.load_and_explore_data(self.test_data_path)
        X, y = self.pipeline.prepare_features()
        self.pipeline.split_data(X, y)
        self.pipeline.scale_features()
        self.pipeline.train_baseline_model()
        self.pipeline.perform_cross_validation()
        
        comparison_df, best_model_name = self.pipeline.compare_and_select_models()
        
        # Check comparison DataFrame
        self.assertIsInstance(comparison_df, pd.DataFrame)
        self.assertGreater(len(comparison_df), 0)
        
        # Check required columns
        required_columns = ['Model', 'Test_AUC_PR', 'Test_F1', 'CV_AUC_PR_Mean', 'CV_AUC_PR_Std']
        for col in required_columns:
            self.assertIn(col, comparison_df.columns)
        
        # Check best model selection
        self.assertIsInstance(best_model_name, str)
        self.assertIn(best_model_name, self.pipeline.models)


class TestDataValidation(unittest.TestCase):
    """Test cases for data validation and error handling."""
    
    def test_missing_target_column(self):
        """Test handling of missing target column."""
        # Create data without target column
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        
        test_path = 'test_no_target.csv'
        df.to_csv(test_path, index=False)
        
        pipeline = FraudDetectionPipeline()
        
        with self.assertRaises(ValueError):
            pipeline.load_and_explore_data(test_path)
        
        # Clean up
        os.remove(test_path)
    
    def test_empty_dataset(self):
        """Test handling of empty dataset."""
        # Create empty dataset
        df = pd.DataFrame(columns=['feature1', 'Class'])
        test_path = 'test_empty.csv'
        df.to_csv(test_path, index=False)
        
        pipeline = FraudDetectionPipeline()
        
        # Should handle gracefully or raise appropriate error
        try:
            pipeline.load_and_explore_data(test_path)
        except Exception as e:
            self.assertIsInstance(e, (ValueError, IndexError))
        
        # Clean up
        os.remove(test_path)
    
    def test_nonexistent_file(self):
        """Test handling of nonexistent file."""
        pipeline = FraudDetectionPipeline()
        
        with self.assertRaises(FileNotFoundError):
            pipeline.load_and_explore_data('nonexistent_file.csv')


class TestModelPerformance(unittest.TestCase):
    """Test cases for model performance validation."""
    
    def setUp(self):
        """Set up test data."""
        # Create a more challenging dataset
        np.random.seed(42)
        n_samples = 2000
        n_fraud = 60  # 3% fraud rate
        n_normal = n_samples - n_fraud
        
        data = []
        
        # Normal transactions with more realistic patterns
        for i in range(n_normal):
            data.append({
                'Amount': max(np.random.lognormal(3, 1), 1),
                'V1': np.random.normal(0, 1),
                'V2': np.random.normal(0, 1),
                'V3': np.random.normal(0, 1),
                'V4': np.random.normal(0, 1),
                'V5': np.random.normal(0, 1),
                'Class': 0
            })
        
        # Fraud transactions with distinct patterns
        for i in range(n_fraud):
            data.append({
                'Amount': max(np.random.lognormal(5, 1.5), 1),
                'V1': np.random.normal(3, 1),    # More separated
                'V2': np.random.normal(-2, 1),   # More separated
                'V3': np.random.normal(2, 1),    # More separated
                'V4': np.random.normal(-3, 1),   # More separated
                'V5': np.random.normal(2.5, 1),  # More separated
                'Class': 1
            })
        
        self.test_df = pd.DataFrame(data).sample(frac=1, random_state=42).reset_index(drop=True)
        self.test_data_path = 'test_performance_data.csv'
        self.test_df.to_csv(self.test_data_path, index=False)
    
    def tearDown(self):
        """Clean up test data."""
        if os.path.exists(self.test_data_path):
            os.remove(self.test_data_path)
    
    def test_minimum_performance_threshold(self):
        """Test that models meet minimum performance thresholds."""
        pipeline = FraudDetectionPipeline(random_state=42)
        results = pipeline.run_complete_pipeline(self.test_data_path)
        
        best_results = results['results'][results['best_model_name']]
        
        # Minimum performance thresholds for synthetic data
        self.assertGreater(best_results['auc_pr'], 0.5, "AUC-PR should be better than random")
        self.assertGreater(best_results['f1_score'], 0.1, "F1-score should be reasonable")
        self.assertGreater(best_results['recall'], 0.1, "Should catch some fraud cases")
        self.assertGreater(best_results['precision'], 0.1, "Should have some precision")
    
    def test_model_consistency(self):
        """Test model consistency across runs."""
        # Run pipeline twice with same random state
        pipeline1 = FraudDetectionPipeline(random_state=42)
        results1 = pipeline1.run_complete_pipeline(self.test_data_path)
        
        pipeline2 = FraudDetectionPipeline(random_state=42)
        results2 = pipeline2.run_complete_pipeline(self.test_data_path)
        
        # Results should be identical with same random state
        best_results1 = results1['results'][results1['best_model_name']]
        best_results2 = results2['results'][results2['best_model_name']]
        
        self.assertAlmostEqual(best_results1['auc_pr'], best_results2['auc_pr'], places=3)
        self.assertAlmostEqual(best_results1['f1_score'], best_results2['f1_score'], places=3)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestFraudDetectionPipeline))
    test_suite.addTest(unittest.makeSuite(TestAdvancedFraudDetectionPipeline))
    test_suite.addTest(unittest.makeSuite(TestDataValidation))
    test_suite.addTest(unittest.makeSuite(TestModelPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}")