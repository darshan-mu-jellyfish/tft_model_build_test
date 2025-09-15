# Use official Python 3.11 slim base image
FROM python:3.11-slim

# ------------------------------
# System dependencies
# ------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential \
        curl \
        wget \
        unzip \
        ffmpeg \
        libsndfile1 \
        && rm -rf /var/lib/apt/lists/*

# ------------------------------
# Set working directory
# ------------------------------
WORKDIR /app

# ------------------------------
# Copy repo files
# ------------------------------
COPY ./app ./app
COPY pipeline_forecast.py ./pipeline_forecast.py
COPY tft_pipeline.py ./tft_pipeline.py
COPY compile_and_submit.py ./compile_and_submit.py
COPY requirements.txt ./requirements.txt

# ------------------------------
# Install Python dependencies
# ------------------------------
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Add Darts extras for PyTorch backend
RUN pip install --no-cache-dir "u8darts[torch]"

# ------------------------------
# Entrypoint
# ------------------------------
ENTRYPOINT ["python", "pipeline_forecast.py"]
