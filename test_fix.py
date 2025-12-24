from src.data_science.engineer import FeatureEngineer
from src.xai.shap_explainer import SHAPExplainer
import pandas as pd
import yaml

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

engineer = FeatureEngineer(config)
explainer = SHAPExplainer()

# Mock request data
input_dict = {
    "person_age": 20,
    "person_gender": "Male",
    "person_income": 10000,
    "person_home_ownership": "RENT",
    "person_emp_length": 0,
    "loan_intent": "EDUCATION",
    "loan_grade": "B",
    "loan_amnt": 1000000,
    "loan_int_rate": 11.25,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 2
}

df_raw = pd.DataFrame([input_dict])
df_proc = engineer.process_pipeline(df_raw)

print("Processed Columns (before drop):", df_proc.columns.tolist())
if 'person_gender' in df_proc.columns:
    df_instance = df_proc.drop(columns=['person_gender'])
else:
    df_instance = df_proc
print("Instance Columns (after drop):", df_instance.columns.tolist())
print("Number of columns:", len(df_instance.columns))

# Test SHAP
explanation = explainer.explain_instance(df_proc)
print("Probability:", explanation['prediction_prob'])
print("Success!")
