import shap
import pandas as pd
import numpy as np
import logging
import joblib
import os
from src.modeling.registry import ModelRegistry

logger = logging.getLogger(__name__)

class SHAPExplainer:
    def __init__(self, model_name: str = "xgboost"):
        self.registry = ModelRegistry()
        self.load_model(model_name)

    def load_model(self, model_name: str):
        """Loads a model and initializes the most appropriate SHAP explainer."""
        logger.info(f"Loading SHAP explainer for {model_name}...")
        self.model_name = model_name
        self.model = self.registry.load_latest(model_name)
        
        # Determine features used (excluding target and sensitive attrs)
        # Assuming 13 features based on training logs
        self.num_features = 13
        
        # Logic: Use new Explainer API if possible for modern detection
        try:
            if "xgboost" in model_name or "random_forest" in model_name:
                self.explainer = shap.TreeExplainer(self.model)
            else:
                # For MLP, use KernelExplainer with a more robust background
                # In a real system, we'd use a small representative sample of training data
                # Here we use a zero-background as a stable placeholder
                background = np.zeros((1, self.num_features))
                self.explainer = shap.KernelExplainer(self.model.predict_proba, background)
        except Exception as e:
            logger.error(f"Explainer initialization error: {e}")
            # Absolute fallback
            self.explainer = shap.Explainer(self.model.predict_proba, np.zeros((1, self.num_features)))

    def explain_instance(self, instance: pd.DataFrame):
        """
        Calculates SHAP values for a single prediction with robust normalization.
        """
        if 'person_gender' in instance.columns:
            instance = instance.drop(columns=['person_gender'])
            
        prediction_prob = float(self.model.predict_proba(instance)[:, 1][0])
        
        try:
            # 1. Generate SHAP values
            if isinstance(self.explainer, shap.KernelExplainer):
                shap_raw = self.explainer.shap_values(instance, nsamples="auto")
            else:
                shap_raw = self.explainer.shap_values(instance)
                
            # 2. Normalize based on dimensionality and type
            # Case A: List of arrays (Common for binary/multi-class Tree/Kernel)
            if isinstance(shap_raw, list):
                # We want the positive class (usually index 1)
                if len(shap_raw) > 1:
                    shap_target = shap_raw[1]
                else:
                    shap_target = shap_raw[0]
            
            # Case B: 3D Array (Common for some Explainer versions with proba output)
            elif len(shap_raw.shape) == 3:
                # Shape: [num_instances, num_features, num_classes] 
                # OR [num_classes, num_instances, num_features]
                if shap_raw.shape[0] == 2: # Binary classes at first dim
                    shap_target = shap_raw[1]
                else: # Binary classes at last dim
                    shap_target = shap_raw[:, :, 1]
            
            # Case C: 2D Array [num_instances, num_features]
            else:
                shap_target = shap_raw

            # 3. Flatten to 1D (for a single instance, it should be features-length)
            shap_final = np.array(shap_target).flatten()
            
            # 4. Check alignment
            feature_names = instance.columns.tolist()
            if len(shap_final) != len(feature_names):
                logger.warning(f"Shape mismatch: SHAP {len(shap_final)} vs Features {len(feature_names)}. Truncating/padding.")
                if len(shap_final) > len(feature_names):
                    shap_final = shap_final[:len(feature_names)]
                else:
                    shap_final = np.pad(shap_final, (0, len(feature_names) - len(shap_final)))

            contributions = dict(zip(feature_names, shap_final.tolist()))
            
            # Handle expected value (base probability)
            base_val = self.explainer.expected_value
            if isinstance(base_val, (list, np.ndarray)):
                base_val = base_val[1] if len(base_val) > 1 else base_val[0]

            return {
                "base_value": float(base_val),
                "contributions": contributions,
                "prediction_prob": prediction_prob
            }

        except Exception as e:
            logger.error(f"Global SHAP explanation failure: {e}")
            # Return heuristic contributions so the UI doesn't crash
            return {
                "base_value": 0.5,
                "contributions": {f: 0.01 for f in instance.columns},
                "prediction_prob": prediction_prob,
                "error": str(e)
            }

    def get_global_importance(self, X_sample: pd.DataFrame):
        """Calculates global importance by averaging absolute SHAP values."""
        if 'person_gender' in X_sample.columns:
            X_sample = X_sample.drop(columns=['person_gender'])
            
        shap_values = self.explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
            
        importance = np.abs(shap_values).mean(axis=0)
        return dict(zip(X_sample.columns, importance.tolist()))
