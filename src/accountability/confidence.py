import numpy as np
import logging

logger = logging.getLogger(__name__)

class ConfidenceEstimator:
    def __init__(self, high_threshold: float = 0.8, low_threshold: float = 0.4):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

    def estimate(self, prob: float):
        """
        Estimates confidence based on the distance from the decision boundary (0.5).
        Also flags low-confidence predictions for human review.
        """
        # Confidence score ranges from 0 to 1
        # |prob - 0.5| * 2
        raw_confidence = abs(prob - 0.5) * 2
        
        status = "High"
        review_required = False
        
        if raw_confidence < self.low_threshold:
            status = "Low - Manual Review Recommended"
            review_required = True
        elif raw_confidence < self.high_threshold:
            status = "Medium"
            
        return {
            "score": float(raw_confidence),
            "status": status,
            "review_required": review_required,
            "reason": "Prediction is near the decision boundary (0.5), indicating uncertainty." if review_required else "Model is confident in its classification."
        }
