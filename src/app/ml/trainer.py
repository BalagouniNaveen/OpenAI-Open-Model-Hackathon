# trainer: creates a simple IsolationForest model from sample data
import joblib
import os
import numpy as np
from sklearn.ensemble import IsolationForest

MODEL_PATH = os.getenv("ML_MODEL_PATH", "/data/detector.pkl")

def train_sample_model(force=False):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if os.path.exists(MODEL_PATH) and not force:
        return MODEL_PATH
    # create synthetic training data: [value, type_code]
    rng = np.random.RandomState(42)
    temps = 20 + 3 * rng.randn(200, 1)  # temperature cluster around 20
    motions = rng.randint(0,2,(200,1))  # motion 0/1
    data = np.vstack([np.hstack([temps, np.zeros((200,1))]),
                      np.hstack([20 + rng.randn(200,1)*5, np.ones((200,1))])])
    # shape: many samples
    X = np.vstack([temps, np.hstack([20 + rng.randn(200,1)*5, np.ones((200,1))])])
    X = np.vstack([np.hstack([temps, np.zeros((temps.shape[0],1))]) , np.hstack([temps+0.5, np.ones((temps.shape[0],1))])])
    X = np.vstack([X, 18 + 2*rng.randn(200,1)])
    # For simplicity create features: [value, type_code]
    # We'll create a simple synthetic matrix:
    values = np.concatenate([20 + 2*rng.randn(400), 18 + 3*rng.randn(200)])
    types = np.concatenate([np.zeros(400), np.ones(200)])
    X = np.column_stack([values, types])
    clf = IsolationForest(random_state=42, contamination=0.05)
    clf.fit(X)
    joblib.dump(clf, MODEL_PATH)
    return MODEL_PATH
