import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class CounterfactualEngine:
    def __init__(self, model, engineer):
        self.model = model
        self.engineer = engineer

    def find_path_to_approval(self, raw_input: dict, steps: int = 10):
        """
        Heuristically finds what changes would lead to approval.
        Focuses on: loan_amnt, person_income, person_emp_length.
        """
        base_df = pd.DataFrame([raw_input])
        base_prob = self.model.predict_proba(self.engineer.process_pipeline(base_df))[:, 1][0]
        
        if base_prob <= 0.5:
            return {"message": "Already approved", "suggestion": None}

        recommendations = []
        
        # 1. Strategy: Reduce Loan Amount
        loan_step = raw_input['loan_amnt'] / steps
        for i in range(1, steps + 1):
            modified = raw_input.copy()
            modified['loan_amnt'] = max(0, raw_input['loan_amnt'] - (loan_step * i))
            prob = self.model.predict_proba(self.engineer.process_pipeline(pd.DataFrame([modified])))[:, 1][0]
            if prob <= 0.5:
                recommendations.append({
                    "feature": "Loan Amount",
                    "current": raw_input['loan_amnt'],
                    "suggested": modified['loan_amnt'],
                    "improvement": f"Reduce loan by ${raw_input['loan_amnt'] - modified['loan_amnt']:,.0f}",
                    "new_prob": float(prob)
                })
                break

        # 2. Strategy: Increase Income
        income_step = raw_input['person_income'] * 0.1 # 10% steps
        for i in range(1, steps + 1):
            modified = raw_input.copy()
            modified['person_income'] = raw_input['person_income'] + (income_step * i)
            prob = self.model.predict_proba(self.engineer.process_pipeline(pd.DataFrame([modified])))[:, 1][0]
            if prob <= 0.5:
                recommendations.append({
                    "feature": "Annual Income",
                    "current": raw_input['person_income'],
                    "suggested": modified['person_income'],
                    "improvement": f"Increase income to ${modified['person_income']:,.0f}",
                    "new_prob": float(prob)
                })
                break

        return {
            "current_prob": float(base_prob),
            "recommendations": recommendations,
            "can_be_approved": len(recommendations) > 0
        }
