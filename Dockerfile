FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY serve.py .
COPY utils/ ./utils/
# Do not copy models/logs into the image so they can be mounted from the host at runtime
RUN mkdir -p /app/models /app/logs
VOLUME ["/app/models", "/app/logs"]

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]