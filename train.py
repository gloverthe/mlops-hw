"""
train.py

Train a simple logistic regression model on the Iris dataset, log parameters and
metrics to MLflow, and persist the trained model to disk.

Environment variables:
- LOGS_DIR: directory for log files (default: "logs")
- RANDOM_SEED: random seed for reproducibility (default: 42)
- MODEL_NAME: base name for saved model files (default: "model")
- MODEL_DIR: directory where model files are saved (default: "models")
- MLFLOW_MLRUNS_PATH: local mlruns directory used by MLflow (default: "mlruns")
- MLFLOW_EXPERIMENT_NAME: MLflow experiment name (default: "default_experiment")

Usage:
    python train.py
"""

import os
import logging
import joblib
import mlflow

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from utils.logger import get_logger

import dotenv

# Load environment variables from a .env file if present
dotenv.load_dotenv()

# Configuration from environment (with sensible defaults)
LOGS_DIR = os.getenv("LOGS_DIR", "logs")
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
MODEL_NAME = os.getenv("MODEL_NAME", "model")
MODEL_DIR = os.getenv("MODEL_DIR", "models")

# Configure MLflow to use a local file-based tracking URI and experiment name
mlruns_path = os.path.abspath(os.getenv("MLFLOW_MLRUNS_PATH", "mlruns"))
mlflow.set_tracking_uri(f"file://{mlruns_path}")
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "default_experiment"))

# Set up logger that writes to a file inside LOGS_DIR
logger = get_logger(
    name="train_logger", log_file=os.path.join(LOGS_DIR, "training.log")
)

# Ensure model output directory exists
os.makedirs(MODEL_DIR, exist_ok=True)


def train():
    """
    Train a LogisticRegression model on the Iris dataset, log training parameters
    and accuracy to MLflow, save the trained model to disk, and log key events.

    Steps:
    1. Load Iris dataset and split into train/test.
    2. Define model hyperparameters and start an MLflow run.
    3. Fit the model, compute test accuracy, and log metrics/params to MLflow.
    4. Persist the model using joblib and log the saved artifact to MLflow.
    """
    # Load dataset and split into train/test sets
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    # Model hyperparameters
    params = {"C": 1.0, "solver": "lbfgs", "max_iter": 200, "random_state": RANDOM_SEED}

    # Start an MLflow run to log params, metrics and artifacts
    with mlflow.start_run():
        mlflow.log_params(params)

        # Initialize and train the model
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        # Evaluate on the test set and log accuracy
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        mlflow.log_metric("accuracy", acc)

        # Get MLflow run id for naming/saving artifacts
        run_id = mlflow.active_run().info.run_id

        # Save model to disk and log as an MLflow artifact
        model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_{run_id}.pkl")
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path, artifact_path="models")

        # Log informational messages about the trained model and run
        logger.info(f"Trained model saved to {model_path}, accuracy={acc:.4f}")
        logger.info(f"MLflow run ID: {run_id}")


if __name__ == "__main__":
    train()
