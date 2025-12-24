import pandas as pd
import numpy as np
import os

def generate_credit_data(n_samples=5000):
    np.random.seed(42)
    
    # Feature generation
    age = np.random.randint(20, 70, n_samples)
    income = np.random.normal(50000, 20000, n_samples).clip(15000, 250000)
    emp_length = np.random.randint(0, 40, n_samples).clip(0, age - 18)
    loan_amnt = np.random.normal(10000, 5000, n_samples).clip(500, 35000)
    loan_int_rate = np.random.normal(11, 3, n_samples).clip(5, 25)
    cred_hist_length = (np.random.randint(2, 30, n_samples)).clip(2, age - 18)
    
    home_ownership = np.random.choice(['RENT', 'OWN', 'MORTGAGE', 'OTHER'], n_samples)
    loan_intent = np.random.choice(['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'], n_samples)
    loan_grade = np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'], n_samples, p=[0.2, 0.3, 0.2, 0.15, 0.08, 0.05, 0.02])
    cb_person_default_on_file = np.random.choice(['Y', 'N'], n_samples, p=[0.15, 0.85])
    
    # Simulate gender for bias analysis demonstration
    gender = np.random.choice(['Male', 'Female'], n_samples)

    # Calculate loan_percent_income
    loan_percent_income = loan_amnt / income
    
    # Logic for ground truth (loan_status)
    # High risk if: low income, high loan amount, high interest, previous default, or low grade
    risk_score = (
        (loan_percent_income * 2) + 
        (loan_int_rate / 10) + 
        (np.where(cb_person_default_on_file == 'Y', 1, 0)) +
        (np.where(loan_grade >= 'D', 1.5, 0)) -
        (income / 100000)
    )
    
    # Add some noise
    risk_score += np.random.normal(0, 0.5, n_samples)
    
    # Convert to binary status (1 = Default/Risk, 0 = Safe)
    loan_status = (risk_score > 1.5).astype(int)
    
    df = pd.DataFrame({
        'person_age': age,
        'person_gender': gender,
        'person_income': income,
        'person_home_ownership': home_ownership,
        'person_emp_length': emp_length,
        'loan_intent': loan_intent,
        'loan_grade': loan_grade,
        'loan_amnt': loan_amnt,
        'loan_int_rate': loan_int_rate,
        'loan_status': loan_status,
        'loan_percent_income': loan_percent_income,
        'cb_person_default_on_file': cb_person_default_on_file,
        'cb_person_cred_hist_length': cred_hist_length
    })
    
    # Invalidate some data for cleaning demonstration
    df.loc[np.random.choice(df.index, 50), 'person_emp_length'] = np.nan
    df.loc[np.random.choice(df.index, 20), 'person_age'] = 120 # Outlier
    
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/credit_risk_dataset.csv', index=False)
    print(f"Generated {n_samples} samples and saved to data/raw/credit_risk_dataset.csv")

if __name__ == "__main__":
    generate_credit_data()
