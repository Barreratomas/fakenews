# 1. Base Image: Python 3.10 Slim (balance size/compatibility)
FROM python:3.10-slim

# 2. Set Environment Variables
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing pyc files to disc
# PYTHONUNBUFFERED: Prevents Python from buffering stdout and stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models/deberta_lora \
    PYTHONPATH=/app

# 3. Set Working Directory
WORKDIR /app

# 4. Install System Dependencies
# gcc and python3-dev might be needed for some python packages like sentencepiece or newspaper3k dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libxml2-dev \
    libxslt-dev \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# 5. Install Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy Application Code
COPY . .

# 7. Create necessary directories for models if they don't exist (handled by COPY but good practice)
# We assume models are in the build context based on .dockerignore rules

# 8. Expose Ports
# 8000 for FastAPI, 7860 for Gradio
EXPOSE 8000 7860

# 9. Command to Run
# We launch Gradio as the primary entrypoint for the demo.
# Note: src.ui.gradio_app will run the UI.
CMD ["python", "src/ui/gradio_app.py"]
