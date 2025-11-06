# Makefile

.PHONY: train evaluate predict clean

train:
	python train.py

metrics:
	python read_mlflow_metrics.py

serve:
	python serve.py

clean:
	rm -f models/*.pkl
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf tests/__pycache__