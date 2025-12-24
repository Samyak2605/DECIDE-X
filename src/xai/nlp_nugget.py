import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class NLPNugget:
    """
    Generates human-readable narratives from SHAP values.
    Mimics GenAI output using structured templates for reliability and local execution.
    """
    
    FEATURE_FRIENDLY_NAMES = {
        "person_age": "age",
        "person_income": "annual income",
        "person_emp_length": "length of employment",
        "loan_amnt": "requested loan amount",
        "loan_int_rate": "interest rate",
        "loan_percent_income": "ratio of loan to income",
        "cb_person_cred_hist_length": "credit history length",
        "loan_to_income": "debt-to-income ratio",
        "stability_index": "employment stability",
        "person_home_ownership": "home ownership status",
        "loan_grade": "credit grade",
        "cb_person_default_on_file": "prior default history"
    }

    def generate_narrative(self, explanation_data: Dict, tone: str = "executive"):
        contributions = explanation_data['contributions']
        prob = explanation_data['prediction_prob']
        
        # Sort contributions by magnitude
        sorted_feats = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        top_positive = [f[0] for f in sorted_feats if f[1] > 0][:2]
        top_negative = [f[0] for f in sorted_feats if f[1] < 0][:2]
        
        is_denied = prob > 0.5
        decision_str = "REJECTED" if is_denied else "APPROVED"
        
        narrative = []
        
        if tone == "technical":
            # Technical Tone: Focus on SHAP values and log-odds
            narrative.append(f"Model output: {decision_str}. The log-odds contribution is dominated by {sorted_feats[0][0]} ({sorted_feats[0][1]:.3f}).")
            narrative.append(f"Significant variance detected in {', '.join([f[0] for f in sorted_feats[:3]])}.")
        elif tone == "simple":
            # Simple Tone: Everyday language
            if is_denied:
                narrative.append("We couldn't approve your request at this time.")
                narrative.append(f"Main reasons: Your {self.FEATURE_FRIENDLY_NAMES.get(top_positive[0], top_positive[0])} and {self.FEATURE_FRIENDLY_NAMES.get(top_positive[1], top_positive[1])} are higher than typical safe limits.")
            else:
                narrative.append("Good news! Your loan application is approved.")
                narrative.append(f"Your strong {self.FEATURE_FRIENDLY_NAMES.get(top_negative[0], top_negative[0])} made the difference.")
        else:
            # Executive Tone (Default): Professional analyst style
            if is_denied:
                reasons = [self.FEATURE_FRIENDLY_NAMES.get(f, f) for f in top_positive]
                narrative.append(f"The analysis indicates a **higher risk profile** primarily driven by {reasons[0]} and {reasons[1]}.")
            else:
                strengths = [self.FEATURE_FRIENDLY_NAMES.get(f, f) for f in top_negative]
                narrative.append(f"This application is marked as **favorable** due to strong indicators in {strengths[0]} and {strengths[1]}.")

        # Level 3: Add Caveats & Honesty
        narrative.append("\n\n*Note: This explanation is based on historical patterns and should be used as a decision-support tool, not a final verdict.*")

        return {
            "verdict": f"Status: {decision_str}",
            "narrative": " ".join(narrative),
            "top_positive": top_positive,
            "top_negative": top_negative
        }
