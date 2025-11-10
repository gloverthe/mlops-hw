"""
serve.py

Simple FastAPI service to load a scikit-learn model and expose a /predict endpoint.

Behavior:
- On startup, loads a model file from MODEL_DIR. If MODEL_RUN_ID == "latest",
    picks the most recently modified file matching MODEL_NAME_*.pkl, or falls back
    to MODEL_NAME.pkl or MODEL_NAME_latest.pkl.
- Exposes POST /predict which accepts a JSON body with "instances": List[List[float]]
    and returns predictions, basic input statistics, and latency.

Environment variables:
- LOGS_DIR (default "logs")
- MODEL_DIR (default "models")
- MODEL_NAME (default "model")
- MODEL_RUN_ID (default "latest")
"""

import time
import logging
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
import glob
import uvicorn

import dotenv

dotenv.load_dotenv()

from utils.logger import get_logger

LOGS_DIR = os.getenv("LOGS_DIR", "logs")
MODEL_DIR = os.getenv("MODEL_DIR", "models")
MODEL_NAME = os.getenv("MODEL_NAME", "model")
MODEL_RUN_ID = os.getenv("MODEL_RUN_ID", "latest")

# initialize application logger (writes to LOGS_DIR/api.log)
logger = get_logger(name="api_logger", log_file=os.path.join(LOGS_DIR, "api.log"))


class PredictRequest(BaseModel):
    """
    Request model for /predict endpoint.

    Attributes:
            instances: A list of feature vectors (each a list of floats). Example:
                                    {"instances": [[0.1, 1.2, 3.4], [0.5, 1.6, 3.8]]}
    """
    instances: List[List[float]]


app = FastAPI()


@app.on_event("startup")
def load_model():
    """
    Load the ML model into app.state.model on startup.

    Loading logic:
    - If MODEL_RUN_ID == "latest", try to find the most recently modified file
        matching MODEL_NAME_*.pkl under MODEL_DIR.
    - If no candidates are found, fall back to MODEL_NAME.pkl or MODEL_NAME_latest.pkl.
    - If MODEL_RUN_ID is set to a specific id, load MODEL_NAME_{MODEL_RUN_ID}.pkl.
    - Any exception during loading is logged and re-raised to fail startup.
    """
    try:
            # Determine which model file to load based on MODEL_RUN_ID
            if MODEL_RUN_ID == "latest":
                    pattern = f"{MODEL_DIR}/{MODEL_NAME}_*.pkl"
                    candidates = glob.glob(pattern)
                    if candidates:
                            # pick the most recently modified candidate
                            MODEL_TO_LOAD = max(candidates, key=lambda p: os.path.getmtime(p))
                    else:
                            # fallback to an unversioned or a *_latest filename
                            fallback = f"{MODEL_DIR}/{MODEL_NAME}.pkl"
                            MODEL_TO_LOAD = (
                                    fallback
                                    if os.path.exists(fallback)
                                    else f"{MODEL_DIR}/{MODEL_NAME}_latest.pkl"
                            )
            else:
                    # specific run id provided
                    MODEL_TO_LOAD = f"{MODEL_DIR}/{MODEL_NAME}_{MODEL_RUN_ID}.pkl"

            logger.info(f"Model to load: {MODEL_TO_LOAD}")
            logger.info(f"Logs directory: {LOGS_DIR}")
            logger.info(f"Model directory: {MODEL_DIR}")

            # load model (expects a joblib-serialised scikit-learn-like estimator)
            app.state.model = joblib.load(MODEL_TO_LOAD)
            logger.info(f"Model name {MODEL_TO_LOAD} loaded successfully")
    except Exception as e:
            # log the exception with stack trace and re-raise to stop the app from starting
            logger.exception("Failed to load model: %s", MODEL_TO_LOAD)
            raise


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Predict endpoint.

    Input:
            JSON body matching PredictRequest (instances: List[List[float]])

    Output:
            JSON containing:
                - predictions: list of model predictions
                - stats: basic input statistics (count, mean, std)
                - latency: request handling time in seconds

    Errors:
            Returns HTTP 500 if the model prediction fails.
    """
    start = time.time()

    # convert the incoming data to a numpy array for model consumption
    X = np.array(req.instances)

    # compute simple aggregate statistics about the input
    stats = {
            "count": int(X.shape[0]),
            "mean": X.mean().item(),
            "std": X.std().item(),
    }

    try:
            # perform prediction using the loaded model
            preds = app.state.model.predict(X).tolist()
    except Exception as e:
            # log the failure and return a 500 to the client
            logger.exception(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    latency = time.time() - start

    # prepare a structured log entry for observability
    log_entry = {
            "count": stats["count"],
            "mean": stats["mean"],
            "std": stats["std"],
            "latency": latency,
            "preds": preds,
    }

    # Use json.dumps to ensure valid JSON format with double quotes
    import json

    logger.info(json.dumps(log_entry))

    # return prediction payload
    return {"predictions": preds, "stats": stats, "latency": latency}


if __name__ == "__main__":
    # start uvicorn when running python serve.py directly. reload=True is useful
    # for local development to auto-restart on code changes.
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=True)
