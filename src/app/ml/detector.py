import os
import joblib
import numpy as np
from .trainer import train_sample_model, MODEL_PATH

MODEL_PATH = os.getenv("ML_MODEL_PATH", MODEL_PATH)

def _load_model():
    if not os.path.exists(MODEL_PATH):
        train_sample_model()
    return joblib.load(MODEL_PATH)

_model = None

def init_detector():
    global _model
    _model = _load_model()

def predict(sensor_type: str, value: float):
    """
    returns: (is_anomaly: bool, score: float)
    uses features [value, type_code] where type_code: temperature=0, motion=1, custom=2
    """
    global _model
    if _model is None:
        init_detector()
    type_map = {"temperature": 0.0, "motion": 1.0}
    tcode = type_map.get(sensor_type, 2.0)
    X = np.array([[float(value), float(tcode)]])
    # IsolationForest: decision_function -> anomaly score (higher is normal); predict -> 1 normal, -1 outlier
    score = float(_model.decision_function(X)[0])
    pred = int(_model.predict(X)[0])
    is_anomaly = (pred == -1)
    # normalize score (rough)
    return is_anomaly, score
