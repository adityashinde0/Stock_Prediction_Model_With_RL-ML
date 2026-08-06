# Use official lightweight Python 3.11 runtime
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Prevent Python from writing bytecode (.pyc) and buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Tell Gradio to listen on all network interfaces
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"

# Install Linux system dependencies required for OpenCV and EasyOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .

# Install PyTorch CPU-only first to save container image size, then install remaining requirements
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose default Gradio port
EXPOSE 7860

# Launch application
CMD ["python", "app.py"]