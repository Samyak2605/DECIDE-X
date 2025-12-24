import json
import datetime
import os
import uuid
import logging

logger = logging.getLogger(__name__)

class GovernanceAuditor:
    def __init__(self, log_path: str = "logs/audit_log.json"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w') as f:
                json.dump([], f)

    def log_decision(self, request_data: dict, response_data: dict, model_version: str):
        """Persists a record of the decision for compliance and debugging."""
        audit_entry = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.now().isoformat(),
            "model_version": model_version,
            "input": request_data,
            "output": {
                "prediction": response_data.get("prediction"),
                "probability": response_data.get("probability"),
                "certainty": response_data.get("confidence_score")
            }
        }
        
        try:
            with open(self.log_path, 'r+') as f:
                logs = json.load(f)
                logs.append(audit_entry)
                f.seek(0)
                json.dump(logs[-100:], f, indent=2) # Keep last 100 for performance
            logger.info(f"Audit entry created: {audit_entry['id']}")
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")

    def get_version_changelog(self):
        """Returns dummy changelog for demonstration."""
        return [
            {"version": "v1.3", "date": "2025-12-23", "changes": "Added OOD detection and Multi-model support"},
            {"version": "v1.2", "date": "2025-12-22", "changes": "Optimized feature engineering for inference latency"},
            {"version": "v1.1", "date": "2025-12-21", "changes": "Initial bias auditing integrated"}
        ]
