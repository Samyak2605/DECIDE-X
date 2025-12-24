from src.data_science.loader import DataLoader
from src.data_science.validator import DataValidator
from src.data_science.engineer import FeatureEngineer
import logging
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EDA_Runner")

def run_data_audit():
    loader = DataLoader("config/config.yaml")
    df_raw = loader.load_raw_data()
    
    validator = DataValidator(loader.config)
    engineer = FeatureEngineer(loader.config)
    
    # Audit Raw Data
    logger.info("--- AUDITING RAW DATA ---")
    validator.validate_schema(df_raw)
    quality_report = validator.run_quality_checks(df_raw)
    summary_stats = validator.generate_summary_stats(df_raw)
    
    # Run Engineering Pipeline
    logger.info("--- RUNNING FEATURE ENGINEERING ---")
    df_processed = engineer.process_pipeline(df_raw, is_training=True)
    
    # Save statistics for the UI/Evaluation
    audit_results = {
        "quality_report": quality_report,
        "summary_stats": summary_stats,
        "final_features": list(df_processed.columns)
    }
    
    os.makedirs('data/processed', exist_ok=True)
    with open('data/processed/audit_report.json', 'w') as f:
        json.dump(audit_results, f, indent=4)
        
    df_processed.to_csv('data/processed/cleaned_risk_data.csv', index=False)
    logger.info("EDA and Preprocessing complete. Data ready for modeling.")
    
    print("\n--- Data Insights Summary ---")
    print(f"Target distribution (Loan Default): {quality_report['target_imbalance']}")
    print(f"Engineered Features: {len(df_processed.columns)}")
    print(f"Outliers managed in 'person_age': {quality_report['outliers'].get('person_age', 0)}")

if __name__ == "__main__":
    run_data_audit()
