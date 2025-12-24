import pandas as pd
import yaml
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        
    def _load_config(self, path: str):
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {path}: {e}")
            raise

    def load_raw_data(self) -> pd.DataFrame:
        raw_path = self.config['data']['raw_path']
        if not os.path.exists(raw_path):
            logger.error(f"Raw data file not found at {raw_path}")
            raise FileNotFoundError(f"Raw data file not found at {raw_path}")
        
        logger.info(f"Loading raw data from {raw_path}")
        df = pd.read_csv(raw_path)
        logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns.")
        return df

    def get_feature_lists(self):
        return (
            self.config['data']['numerical_features'],
            self.config['data']['categorical_features'],
            self.config['data']['target']
        )
