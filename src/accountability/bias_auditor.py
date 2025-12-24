import pandas as pd
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class BiasAuditor:
    def __init__(self, sensitive_feature: str = "person_gender"):
        self.sensitive_feature = sensitive_feature

    def calculate_disparate_impact(self, X_test: pd.DataFrame, y_prob: np.ndarray, threshold: float = 0.5):
        """
        Calculates Disparate Impact (DI) - The ratio of positive outcomes (approvals) 
        between protected and unprotected groups.
        DI = (Approval Rate of Group A) / (Approval Rate of Group B)
        Acceptable: DI > 0.8 (The 80% Rule)
        Note: In our case, 'Approval' is y_pred = 0.
        """
        df = X_test.copy()
        df['prediction'] = (y_prob > threshold).astype(int)
        
        # Invert prediction for 'Approval' rate (0=Approved, 1=Denied)
        df['is_approved'] = 1 - df['prediction']
        
        groups = df.groupby(self.sensitive_feature)['is_approved'].mean().to_dict()
        
        if len(groups) < 2:
            return {"status": "insufficient_data", "groups": groups}
            
        group_names = list(groups.keys())
        ratios = {}
        
        for g1 in group_names:
            for g2 in group_names:
                if g1 != g2:
                    ratio = groups[g1] / groups[g2] if groups[g2] > 0 else 0
                    ratios[f"{g1}_vs_{g2}"] = float(ratio)
        
        # Check for DI violations (< 0.8)
        violation = any(r < 0.8 for r in ratios.values())
        
        return {
            "disparate_impact": ratios,
            "approval_rates": groups,
            "violation": violation,
            "warning": "Potential Bias Detected" if violation else "Fairness Check Passed"
        }
