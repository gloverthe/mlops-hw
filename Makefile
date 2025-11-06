# Makefile

# Configuration variables
MODELS_DIR := $$(pwd)/models
LOGS_DIR := $$(pwd)/logs
MODEL_NAME := iris_model
MODEL_RUN_ID := latest
VERSION := v1

.PHONY: train evaluate predict clean build docker-run docker-stop

train:
	python train.py

metrics:
	python mlflow_metrics.py

serve:
	python serve.py

build:
	docker build -t ml-api:latest .

docker-run: docker-stop
	docker run -d \
	-p 8000:8000 \
	-v "$(MODELS_DIR):/app/models" \
	-v "$(LOGS_DIR):/app/logs" \
	-e MODELS_DIR=/app/models \
	-e LOGS_DIR=/app/logs \
	-e MODEL_DIR=/app/models \
	-e MODEL_NAME=$(MODEL_NAME) \
	-e MODEL_RUN_ID=$(MODEL_RUN_ID) \
    --name ml-api_$(VERSION) \
    ml-api:latest

docker-stop:
	docker stop ml-api_$(VERSION) || true
	docker rm ml-api_$(VERSION) || true

clean:
	rm -f models/*.pkl
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf tests/__pycache__