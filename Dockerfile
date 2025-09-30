FROM python:3.10-slim
WORKDIR /app

# --- FIX: Install required system dependencies for OpenCV and other packages ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    git-lfs && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# COPY Model Weights (best.pt from your successful training run)
COPY best.pt .

COPY . .
EXPOSE 10000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]