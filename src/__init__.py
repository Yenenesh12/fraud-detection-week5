# Fraud Detection System
# Source code package

__version__ = "1.0.0"
__author__ = "AI Assistant"

from .fraud_detection_pipeline import FraudDetectionPipeline
from .fraud_detection_advanced import AdvancedFraudDetectionPipeline

__all__ = [
    'FraudDetectionPipeline',
    'AdvancedFraudDetectionPipeline'
]