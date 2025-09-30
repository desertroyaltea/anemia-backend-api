FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- NEW STEP: Install Git LFS to handle large model weights ---
RUN apt-get update && apt-get install -y git-lfs

# --- COPY Model Weights (best.pt from your successful training run) ---
# NOTE: Place your 'best.pt' file in the same directory as this Dockerfile.
COPY best.pt .

COPY . .
EXPOSE 10000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]