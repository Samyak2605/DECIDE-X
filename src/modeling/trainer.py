import pandas as pd
import numpy as np
import logging
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from src.modeling.registry import ModelRegistry
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.registry = ModelRegistry()
        self.target = self.config['data']['target']

    def prepare_data(self, data_path: str):
        df = pd.read_csv(data_path)
        X = df.drop(columns=[self.target])
        y = df[self.target]
        
        # We drop 'person_gender' for training if we want to simulate fairness awareness
        # but keep it in the test set for auditing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        return X_train, X_test, y_train, y_test

    def train_interpretable(self, X_train, y_train, X_test, y_test):
        """Trains XGBoost as the primary interpretable model."""
        params = self.config['model']['params']
        logger.info(f"Training XGBoost with params: {params}")
        
        # Remove person_gender for actual training to demonstrate fairness approach
        X_tr = X_train.drop(columns=['person_gender'])
        X_te = X_test.drop(columns=['person_gender'])

        model = XGBClassifier(**params)
        model.fit(X_tr, y_train)
        
        y_prob = model.predict_proba(X_te)[:, 1]
        metrics = self._evaluate(y_test, y_prob)
        
        self.registry.save_model(model, "xgboost", metrics, params)
        return model, metrics

    def train_baseline_dl(self, X_train, y_train, X_test, y_test):
        """Trains an MLP for accuracy comparison (Traditional Black-box)."""
        logger.info("Training MLP baseline...")
        
        X_tr = X_train.drop(columns=['person_gender'])
        X_te = X_test.drop(columns=['person_gender'])

        model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        model.fit(X_tr, y_train)
        
        y_prob = model.predict_proba(X_te)[:, 1]
        metrics = self._evaluate(y_test, y_prob)
        
        self.registry.save_model(model, "mlp_baseline", metrics, {"hidden_layers": (64, 32)})
        return model, metrics

    def train_rf(self, X_train, y_train, X_test, y_test):
        """Trains Random Forest for model comparison."""
        logger.info("Training Random Forest...")
        
        X_tr = X_train.drop(columns=['person_gender'])
        X_te = X_test.drop(columns=['person_gender'])

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_tr, y_train)
        
        y_prob = model.predict_proba(X_te)[:, 1]
        metrics = self._evaluate(y_test, y_prob)
        
        self.registry.save_model(model, "random_forest", metrics, {"n_estimators": 100})
        return model, metrics

    def _evaluate(self, y_true, y_prob):
        y_pred = (y_prob > 0.5).astype(int)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        
        # Calibration Metrics (Level 1, #2)
        brier = brier_score_loss(y_true, y_prob)
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=5)
        
        metrics = {
            "auc_roc": float(roc_auc_score(y_true, y_prob)),
            "auc_pr": float(auc(recall, precision)),
            "brier_score": float(brier),
            "calibration_curve": {
                "prob_true": prob_true.tolist(),
                "prob_pred": prob_pred.tolist()
            },
            "report": classification_report(y_true, y_pred, output_dict=True)
        }
        logger.info(f"Evaluation Metrics: AUC-ROC: {metrics['auc_roc']:.4f}, Brier: {metrics['brier_score']:.4f}")
        return metrics

if __name__ == "__main__":
    trainer = ModelTrainer("config/config.yaml")
    X_train, X_test, y_train, y_test = trainer.prepare_data("data/processed/cleaned_risk_data.csv")
    
    xgb_model, xgb_metrics = trainer.train_interpretable(X_train, y_train, X_test, y_test)
    mlp_model, mlp_metrics = trainer.train_baseline_dl(X_train, y_train, X_test, y_test)
    rf_model, rf_metrics = trainer.train_rf(X_train, y_train, X_test, y_test)
    
    # Save the test set for auditing and XAI later
    X_test.to_csv("data/processed/test_features.csv", index=False)
    y_test.to_csv("data/processed/test_target.csv", index=False)
    
    print("\n--- Model Comparison ---")
    print(f"XGBoost AUC-ROC: {xgb_metrics['auc_roc']:.4f}")
    print(f"MLP AUC-ROC: {mlp_metrics['auc_roc']:.4f}")
    print(f"Random Forest AUC-ROC: {rf_metrics['auc_roc']:.4f}")
