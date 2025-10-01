# Use an official Python runtime as a parent image
FROM python:3.10-slim

# --- NEW LINE TO INSTALL SYSTEM DEPENDENCIES FOR OPENCV ---
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code (including the models/ folder)
COPY . .

# Make port 10000 available to the world outside this container
EXPOSE 10000

# Run the app directly with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]