import os
import joblib
from app.train import train_model

def test_train_model_creates_file(tmp_path, monkeypatch):
    # Override model dir to temp directory
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))

    # Run training
    train_model()

    # Assert model file exists
    model_file = tmp_path / "model.pkl"
    assert model_file.exists()  # nosec
