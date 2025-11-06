# ML Model Training & API Service

A machine learning application for training an Iris classification model with MLflow tracking and serving predictions via FastAPI.

## 📋 Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Initial Setup](#initial-setup)
- [Environment Variables](#environment-variables)
- [Usage](#usage)
- [Docker Deployment](#docker-deployment)
- [API Endpoints](#api-endpoints)
- [Monitoring](#monitoring)

## ✨ Features

- **Model Training**: Train logistic regression model on Iris dataset
- **MLflow Tracking**: Track experiments, metrics, and model artifacts
- **FastAPI Service**: REST API for model predictions
- **Docker Support**: Containerized deployment with volume mounts
- **Monitoring**: Log parsing and visualization of API metrics
- **Configuration Management**: Environment-based configuration

## 🔧 Prerequisites

- Python 3.9+
- Docker (for containerized deployment)
- pip or conda for package management

## 🚀 Initial Setup

### 1. Clone the Repository

```bash
cd /path/to/mg-mlops-hw
```

### 2. Create Python Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Create Required Directories

```bash
mkdir -p models logs mlruns
```

### 5. Create `.env` File

Create a `.env` file in the project root with the following content:

```bash
# Model Configuration
MODEL_NAME=iris_model
MODEL_RUN_ID=latest
MODEL_DIR=models

# Logging Configuration
LOGS_DIR=logs

# MLflow Configuration
MLFLOW_TRACKING_URI=./mlruns

# Training Configuration
RANDOM_SEED=42

# Docker Configuration (optional)
VERSION=v1
```

**Environment Variable Descriptions:**

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_NAME` | Name prefix for saved model files | `iris_model` |
| `MODEL_RUN_ID` | Specific MLflow run ID or "latest" | `latest` |
| `MODEL_DIR` | Directory to store trained models | `models` |
| `LOGS_DIR` | Directory for application logs | `logs` |
| `MLFLOW_TRACKING_URI` | MLflow tracking directory/server URI | `./mlruns` |
| `RANDOM_SEED` | Random seed for reproducibility | `42` |
| `VERSION` | Docker container version tag | `v1` |

## 📖 Usage

### Makefile Commands

The project includes a Makefile for common operations:

#### **`make train`**
Trains the logistic regression model on the Iris dataset and logs results to MLflow.

```bash
make train
```

**What it does:**
- Loads the Iris dataset
- Splits data into train/test sets
- Trains a Logistic Regression model
- Logs parameters and metrics to MLflow
- Saves model to `models/` directory

#### **`make metrics`**
Reads and displays MLflow experiment metrics.

```bash
make metrics
```

**What it does:**
- Queries MLflow tracking data
- Displays experiment metrics and parameters
- Useful for comparing model runs

#### **`make serve`**
Starts the FastAPI server locally (without Docker).

```bash
make serve
```

**What it does:**
- Loads the latest (or specified) trained model
- Starts FastAPI server on `http://localhost:8000`
- Enables hot-reload for development

**Access the API:**
- Swagger docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

#### **`make build`**
Builds the Docker image for the application.

```bash
make build
```

**What it does:**
- Builds Docker image tagged as `ml-api:latest`
- Installs all dependencies from `requirements.txt`
- Copies application code into the image

#### **`make docker-run`**
Runs the application in a Docker container with volume mounts.

```bash
make docker-run
```

**What it does:**
- Starts a detached Docker container named `ml-api_<VERSION>`
- Exposes API on port `8000`
- Mounts local `models/` and `logs/` directories
- Sets environment variables for configuration
- Container accesses local filesystem for models and logs

**Volume Mounts:**
- `$(pwd)/models:/app/models` - Access trained models
- `$(pwd)/logs:/app/logs` - Write API logs to host

**Environment Variables Passed:**
- `MODELS_DIR`, `LOGS_DIR`, `MODEL_DIR`
- `MODEL_NAME`, `MODEL_RUN_ID`

#### **`make docker-stop`**
Stops and removes the running Docker container.

```bash
make docker-stop
```

**What it does:**
- Stops the container `ml-api_<VERSION>`
- Removes the container (cleanup)

#### **`make clean`**
Cleans up generated files and cache.

```bash
make clean
```

**What it does:**
- Removes all `.pkl` model files from `models/`
- Deletes Python cache directories (`__pycache__`)

## 🐳 Docker Deployment

### Local Development with Docker

```bash
# Build the image
make build

# Run the container
make docker-run

# Check container logs
docker logs -f ml-api_v1

# Stop the container
make docker-stop
```

### Production Deployment

For production, modify the approach to use cloud storage (S3, GCS, etc.) instead of local volume mounts:

1. **Update `serve.py`** to download models from S3/cloud storage at startup
2. **Use environment variables** for cloud credentials (AWS keys, service accounts)
3. **Remove `--reload`** flag from uvicorn command in Dockerfile
4. **Configure logging** to send logs to CloudWatch/ELK/centralized logging


## 🔌 API Endpoints

### `POST /predict`

Make predictions using the trained model.

**Request Body:**
```json
{
  "instances": [
    [5.1, 3.5, 1.4, 0.2],
    [6.2, 3.4, 5.4, 2.3]
  ]
}
```

**Response:**
```json
{
  "predictions": [0, 2],
  "stats": {
    "count": 2,
    "mean": 3.4375,
    "std": 1.9422522364512804
  },
  "latency": 0.000247955
}
```

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"instances": [[5.1, 3.5, 1.4, 0.2]]}'
```

**Example using Python:**

demo.ipynb contains an example of calling the endpoint an viewing the metrics

```python
import requests

url = "http://localhost:8000/predict"
payload = {
    "instances": [
        [5.1, 3.5, 1.4, 0.2],
        [6.2, 3.4, 5.4, 2.3]
    ]
}

response = requests.post(url, json=payload)
print(response.json())
```

## 📊 Monitoring

### Using the Monitor Class

The `monitor.py` module provides visualization of API request metrics:

```python
import monitor
from pathlib import Path

log_file = Path('logs') / 'api.log'
api_monitor = monitor.monitor(log_file=str(log_file))

# Display plots
api_monitor.plot_stats()

# Show summary statistics
api_monitor.display_summary()
```

**Metrics Tracked:**
- Request latency over time
- Input mean values
- Input standard deviation
- Request counts

### Log Format

API logs are written in JSON format to `logs/api.log`:

```
2025-11-06 11:34:28,148 INFO {"count": 2, "mean": 3.4375, "std": 1.9422522364512804, "latency": 0.000247955, "preds": [0, 2]}
```

## 🧪 Testing

### Interactive Testing with Jupyter

Use the provided `demo.ipynb` notebook:

```bash
jupyter notebook demo.ipynb
```

The notebook includes:
- API request examples
- Log parsing and visualization
- Monitoring dashboard

## 🛠️ Troubleshooting

### Issue: MLflow metadata file missing

```bash
# Reset MLflow tracking directory
rm -rf mlruns/
python train.py
```

### Issue: Docker volume mount errors

Ensure paths are absolute and directories exist:
```bash
mkdir -p models logs
```

### Issue: Model file not found

Train a model first or set correct `MODEL_RUN_ID`:
```bash
make train
# Then start the server
make serve
```

### Issue: Port 8000 already in use

```bash
# Find and kill the process
lsof -ti:8000 | xargs kill -9

# Or change the port in serve.py
```

## 📝 Notes

- **Local vs Production**: Volume mounts are for local development. In production, use cloud storage (S3, Azure Blob, GCS).
- **MLflow UI**: Run `mlflow ui` to view experiment tracking dashboard at `http://localhost:5000`
- **Model Versioning**: Each training run creates a new model file with the MLflow run ID suffix

## 📄 License

This project is for educational purposes.

## 🤝 Contributing

1. Create feature branch
2. Make changes
3. Test locally with `make train` and `make serve`
4. Submit pull request

---

**Need help?** Check the API docs at `http://localhost:8000/docs` when the server is running.
