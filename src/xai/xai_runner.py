from src.xai.shap_explainer import SHAPExplainer
from src.xai.nlp_nugget import NLPNugget
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def test_explainer():
    explainer = SHAPExplainer()
    nugget = NLPNugget()
    
    # Load sample test data
    X_test = pd.read_csv("data/processed/test_features.csv")
    sample = X_test.iloc[[0]]
    
    # Explain
    explanation = explainer.explain_instance(sample)
    narrative = nugget.generate_narrative(explanation)
    
    print("\n--- Sample XAI Output ---")
    print(narrative['verdict'])
    print(narrative['narrative'])
    print("\nFeature Contributions (SHAP Values):")
    for k, v in explanation['contributions'].items():
        if abs(v) > 0.1: # Only print major ones
            print(f"- {k}: {v:.4f}")

if __name__ == "__main__":
    test_explainer()
