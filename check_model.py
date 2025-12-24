from src.modeling.registry import ModelRegistry
import joblib
import os

registry = ModelRegistry()
model = registry.load_latest("xgboost")
print("Model Feature Names:", model.get_booster().feature_names)
print("Number of Model Features:", len(model.get_booster().feature_names))
