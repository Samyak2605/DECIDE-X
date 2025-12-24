import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

from sklearn.ensemble import IsolationForest
import joblib
import os

class DataValidator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.thresholds = config.get('thresholds', {})
        self.iso_forest = None
        self.train_baseline = None

    def fit_ood_detector(self, df_train: pd.DataFrame):
        """Fits an OOD detector on training data manifolds."""
        logger.info("Fitting Out-of-Distribution (OOD) detector...")
        features = self.config['data']['numerical_features']
        self.iso_forest = IsolationForest(contamination=0.05, random_state=42)
        self.iso_forest.fit(df_train[features])
        self.train_baseline = df_train[features].mean()
        
        # Save detector path
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.iso_forest, "models/ood_detector.joblib")
        joblib.dump(self.train_baseline, "models/train_baseline.joblib")

    def check_ood(self, instance_df: pd.DataFrame) -> Dict[str, Any]:
        """Checks if a live instance is OOD based on isolation forest score."""
        if self.iso_forest is None:
            if os.path.exists("models/ood_detector.joblib"):
                self.iso_forest = joblib.load("models/ood_detector.joblib")
                self.train_baseline = joblib.load("models/train_baseline.joblib")
            else:
                return {"is_ood": False, "similarity_score": 1.0}

        features = self.config['data']['numerical_features']
        score = self.iso_forest.decision_function(instance_df[features])[0]
        # Map score to a "Similarity Index" (0 to 1)
        # Isolation Forest decision_function returns values in roughly [-0.5, 0.5]
        similarity = 1 / (1 + np.exp(-5 * score)) 
        
        return {
            "is_ood": bool(score < -0.1), # Threshold for OOD
            "similarity_score": round(float(similarity), 3),
            "warning": "Input profile significantly differs from training data" if score < -0.1 else None
        }

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Checks if all required columns exist."""
        required_cols = (
            self.config['data']['numerical_features'] + 
            self.config['data']['categorical_features'] + 
            [self.config['data']['target']]
        )
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            logger.error(f"Schema validation failed. Missing columns: {missing}")
            return False
        logger.info("Schema validation passed.")
        return True

    def run_quality_checks(self, df: pd.DataFrame):
        """Performs missing value analysis and outlier detection."""
        checks = {}
        
        # 1. Missing Values
        missing = df.isnull().sum()
        checks['missing_values'] = missing[missing > 0].to_dict()
        if checks['missing_values']:
            logger.warning(f"Missing values detected: {checks['missing_values']}")
        
        # 2. Outlier Detection (Z-Score for numerical)
        num_features = self.config['data']['numerical_features']
        z_threshold = self.thresholds.get('outlier_z_score', 3.0)
        outliers = {}
        
        for col in num_features:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            num_outliers = (z_scores > z_threshold).sum()
            if num_outliers > 0:
                outliers[col] = int(num_outliers)
        
        checks['outliers'] = outliers
        if outliers:
            logger.warning(f"Potential outliers detected: {outliers}")
            
        # 3. Target Balance
        target = self.config['data']['target']
        counts = df[target].value_counts(normalize=True).to_dict()
        checks['target_imbalance'] = counts
        logger.info(f"Target distribution: {counts}")
        
        return checks

    def generate_summary_stats(self, df: pd.DataFrame):
        """Generates programmatic EDA statistics."""
        summary = {
            "total_rows": len(df),
            "numerical_stats": df.describe().to_dict(),
            "categorical_uniques": {col: df[col].nunique() for col in self.config['data']['categorical_features']}
        }
        logger.info("Generated summary statistics for the dataset.")
        return summary
