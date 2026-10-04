# Use official Python runtime as a parent image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for FAISS and compilation
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY app/ app/
COPY dashboard/ dashboard/
COPY .env.example .env.example

# We don't copy the corpus here because we want to mount it or assume it's mounted
# to simulate "corpus-agnostic" behavior. But for the demo package, we'll copy it.
COPY streaming-rag-dev-demo-corpus/ streaming-rag-dev-demo-corpus/

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
