import joblib
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model/model.pkl")

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

def predict(instances):
    model = load_model()
    return model.predict(np.array(instances)).tolist()
