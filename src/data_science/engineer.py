import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scalers = {}
        self.encoders = {}
        self.target = config['data']['target']
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the dataset by handling missing values and clipping outliers.
        """
        df_clean = df.copy()
        
        # 1. Handle missing emp_length: Use median as it's robust to outliers
        # Risk: Imputing may mask the fact that some users have no employment history
        if 'person_emp_length' in df_clean.columns:
            median_emp = df_clean['person_emp_length'].median()
            df_clean['person_emp_length'] = df_clean['person_emp_length'].fillna(median_emp)
            logger.info(f"Imputed missing person_emp_length with median: {median_emp}")

        # 2. Outlier Clipping: person_age > 100 is likely a data entry error
        # Risk: Clipping can remove genuine extreme data, but in credit risk, 120yr old applicants are errors.
        df_clean['person_age'] = df_clean['person_age'].clip(upper=100)
        
        return df_clean

    def calculate_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates domain-specific features.
        """
        df_feat = df.copy()
        
        # Risk Metric: Percentage of income spent on the loan
        # If not provided (inference time), we calculate it.
        # Use small epsilon to avoid division by zero if income is 0
        if 'loan_percent_income' not in df_feat.columns:
            df_feat['loan_percent_income'] = df_feat['loan_amnt'] / df_feat['person_income'].clip(lower=1)
        
        # Higher debt-to-income is a strong predictor.
        df_feat['loan_to_income'] = df_feat['loan_amnt'] / df_feat['person_income'].clip(lower=1)
        
        # Stability indicator: percent of life spent employed
        df_feat['stability_index'] = df_feat['person_emp_length'] / (df_feat['person_age'] - 16).clip(lower=1)
        
        return df_feat

    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts strings to numerical for tree-based models using Label Encoding.
        """
        df_encoded = df.copy()
        for col in self.config['data']['categorical_features']:
            le = LabelEncoder()
            # Ensure the column exists before encoding
            if col in df_encoded.columns:
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.encoders[col] = le
        return df_encoded

    def process_pipeline(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """Unified pipeline for the data science layer."""
        df_clean = self.clean_data(df)
        df_feat = self.calculate_interactions(df_clean)
        df_final = self.encode_categorical(df_feat)
        
        # Define the exact features used in training (excluding sensitive attributes and target)
        # Numerical (7) + Categorical (4) + Interactions (2) = 13 Features Total
        expected_features = [
            'person_age', 'person_income', 'person_home_ownership', 'person_emp_length',
            'loan_intent', 'loan_grade', 'loan_amnt', 'loan_int_rate', 'loan_percent_income',
            'cb_person_default_on_file', 'cb_person_cred_hist_length', 'loan_to_income',
            'stability_index'
        ]
        
        # If training, we keep the target AND sensitive attributes for auditing
        if is_training:
            if self.target in df_final.columns:
                # We need gender for bias auditing even if not for modeling
                audit_features = ['person_gender'] if 'person_gender' in df_final.columns else []
                df_final = df_final[expected_features + audit_features + [self.target]]
        else:
            # During inference, strictly align with expected feature list
            # Fill missing with zero if any (safety)
            for feat in expected_features:
                if feat not in df_final.columns:
                    df_final[feat] = 0
            df_final = df_final[expected_features]
        
        logger.info("Feature engineering pipeline completed.")
        return df_final
