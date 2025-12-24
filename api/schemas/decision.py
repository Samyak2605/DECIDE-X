from pydantic import BaseModel
from typing import Dict, List, Optional

class DecisionRequest(BaseModel):
    person_age: float
    person_income: float
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: float
    person_gender: Optional[str] = "Male"
    # Level 4: Model Choice
    model_choice: str = "xgboost" 
    # Level 3: Narrative Tone
    tone: str = "executive"

class Contribution(BaseModel):
    feature: str
    value: float

class DecisionResponse(BaseModel):
    prediction: str
    probability: float
    confidence_score: float
    confidence_status: str
    review_required: bool
    narrative: str
    contributions: List[Contribution]
    fairness_warning: str
    
    # Advanced ML Signals
    is_ood: bool = False
    similarity_score: float = 1.0
    counterfactuals: Optional[dict] = None
    brier_score: Optional[float] = None
    
    # Quantitative Ethics
    fairness_metrics: Optional[Dict[str, float]] = None
    
    model_version: str = "v1.3"
