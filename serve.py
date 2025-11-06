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

from load_config import load_config
from utils.logger import get_logger

config = load_config()
logger = get_logger(
    name="api_logger", log_file=os.path.join(config["path_to_logs"], "api.log")
)

# get the required model id
MODEL_NAME = config.get("model_name", "model")
MODEL_RUN_ID = config.get("model_run_id", "latest")
MODEL_PATH = config["path_to_models"]

# if RUN_ID = "latest" pick the newest file matching pattern
if MODEL_RUN_ID == "latest":
    pattern = f"{MODEL_PATH}/{MODEL_NAME}_*.pkl"
    candidates = glob.glob(pattern)
    if candidates:
        MODEL_PATH = max(candidates, key=lambda p: os.path.getmtime(p))
    else:
        # fallback to a simple model.pkl if it exists
        fallback = f"{MODEL_PATH}/{MODEL_NAME}.pkl"
        MODEL_PATH = fallback if os.path.exists(fallback) else f"{MODEL_PATH}/{MODEL_NAME}_latest.pkl"
else:
    MODEL_TO_LOAD = f"{MODEL_PATH}/{MODEL_NAME}_{MODEL_RUN_ID}.pkl"


class PredictRequest(BaseModel):
    instances: List[List[float]]


app = FastAPI()

@app.on_event("startup")
def load_model():
    try:
        app.state.model = joblib.load(MODEL_TO_LOAD)
        logger.info(f"Model name {MODEL_TO_LOAD} loaded successfully")
    except Exception as e:
        logger.exception("Failed to load model: %s", MODEL_TO_LOAD)
        raise

@app.post("/predict")
def predict(req: PredictRequest):
    start = time.time()
    X = np.array(req.instances)
    # simple input stats
    stats = {
        "count": int(X.shape[0]),
        "mean": X.mean().item(),
        "std": X.std().item(),
    }
    try:
        preds = app.state.model.predict(X).tolist()
    except Exception as e:
        logger.exception(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    latency = time.time() - start
    log_entry = {"count": stats["count"], "mean": stats["mean"], "std": stats["std"], "latency": latency, "preds": preds}
    logger.info(log_entry)
    return {"predictions": preds, "stats": stats, "latency": latency}


if __name__ == "__main__":
    # start uvicorn when running python serve.py
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=True)