from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib
import os

def train_model():
    iris = load_iris()
    X, y = iris.data, iris.target

    clf = LogisticRegression(max_iter=200)
    clf.fit(X, y)

    output_dir = os.environ.get("MODEL_DIR", "model")
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model.pkl")

    joblib.dump(clf, model_path)
    print(f"✅ Model saved to {model_path}")

if __name__ == "__main__":
    train_model()
