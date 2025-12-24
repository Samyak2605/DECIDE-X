from src.accountability.bias_auditor import BiasAuditor
from src.accountability.confidence import ConfidenceEstimator
from src.modeling.registry import ModelRegistry
import pandas as pd
import numpy as np

def run_audits():
    registry = ModelRegistry()
    model = registry.load_latest("xgboost")
    
    X_test = pd.read_csv("data/processed/test_features.csv")
    y_test = pd.read_csv("data/processed/test_target.csv")
    
    # Predict
    X_feat = X_test.drop(columns=['person_gender'])
    y_prob = model.predict_proba(X_feat)[:, 1]
    
    # Bias Audit
    auditor = BiasAuditor()
    bias_results = auditor.calculate_disparate_impact(X_test, y_prob)
    
    # Confidence Estimation (on a single point)
    conf_estimator = ConfidenceEstimator()
    sample_prob = y_prob[0]
    conf_results = conf_estimator.estimate(sample_prob)
    
    print("\n--- Bias Audit Results ---")
    print(f"Status: {bias_results['warning']}")
    print(f"Approval Rates by Gender: {bias_results['approval_rates']}")
    print(f"Disparate Impact Ratios: {bias_results['disparate_impact']}")
    
    print("\n--- Confidence Check (Sample) ---")
    print(f"Prob: {sample_prob:.4f}, Confidence: {conf_results['score']:.4f}")
    print(f"Status: {conf_results['status']}")
    print(f"Manual Review: {conf_results['review_required']}")

if __name__ == "__main__":
    run_audits()
