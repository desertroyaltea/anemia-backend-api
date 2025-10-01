# Pin the base image to the stable "Bullseye" version of the OS
FROM python:3.10-slim-bullseye

# Set the environment to be non-interactive to prevent prompts
ENV DEBIAN_FRONTEND=noninteractive

# This command will now work because the package name is correct for this OS version
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code
COPY . .

# Expose the port the app runs on
EXPOSE 10000

# Run the app directly with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]