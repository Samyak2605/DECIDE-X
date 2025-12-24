import joblib
import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(self, base_path: str = "models"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def save_model(self, model, model_name: str, metrics: dict, params: dict):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(self.base_path, f"{model_name}_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, "model.joblib")
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata = {
            "model_name": model_name,
            "timestamp": timestamp,
            "metrics": metrics,
            "params": params
        }
        with open(os.path.join(model_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)
            
        # Update 'latest' pointer
        latest_path = os.path.join(self.base_path, f"{model_name}_latest")
        if os.path.exists(latest_path):
            os.remove(latest_path) if not os.path.isdir(latest_path) else None # Safety
        
        # On some systems symlinks might fail, so we'll just write a pointer file
        with open(os.path.join(self.base_path, f"{model_name}_latest_pointer.txt"), "w") as f:
            f.write(model_dir)
            
        logger.info(f"Model {model_name} saved to {model_dir}")
        return model_dir

    def load_latest(self, model_name: str):
        pointer_path = os.path.join(self.base_path, f"{model_name}_latest_pointer.txt")
        if not os.path.exists(pointer_path):
            logger.error(f"No latest pointer found for {model_name}")
            return None
            
        with open(pointer_path, "r") as f:
            model_dir = f.read().strip()
            
        model = joblib.load(os.path.join(model_dir, "model.joblib"))
        logger.info(f"Loaded latest {model_name} from {model_dir}")
        return model
