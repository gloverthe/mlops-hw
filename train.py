import os
import logging
import joblib
import mlflow

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from utils.logger import get_logger

# from load_config import load_config

# Load configuration
# config = load_config()
import dotenv

dotenv.load_dotenv()

# MODELS_DIR = os.getenv("MODELS_DIR", "models")
LOGS_DIR = os.getenv("LOGS_DIR", "logs")
# CONFIG_FILE = os.getenv("CONFIG_FILE", "config.yaml")

logger = get_logger(
    name="train_logger", log_file=os.path.join(LOGS_DIR, "training.log")
)

RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
MODEL_NAME = os.getenv("MODEL_NAME", "model")
MODEL_DIR = os.getenv("MODEL_DIR", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def train():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    params = {"C": 1.0, "solver": "lbfgs", "max_iter": 200, "random_state": RANDOM_SEED}
    with mlflow.start_run():
        mlflow.log_params(params)
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        mlflow.log_metric("accuracy", acc)

        run_id = mlflow.active_run().info.run_id

        model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_{run_id}.pkl")
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path, artifact_path="models")

        logger.info(f"Trained model saved to {model_path}, accuracy={acc:.4f}")
        logger.info(f"MLflow run ID: {run_id}")


if __name__ == "__main__":
    train()
