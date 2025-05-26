# Use NVIDIA PyTorch image with Python 3.12 and CUDA 12.1
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install extra system dependencies (if needed)
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt /app/requirements.txt

# Copy .env file
COPY .env /app/.env

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install python-dotenv

# Copy the rest of the application code
COPY . /app/

# Set environment variable with default value
ENV ENABLE_MQTT=true

# Expose port
EXPOSE 8000

# Start the server
CMD ["python", "server.py"]
