# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN pip install --no-cache-dir flask faiss-cpu sentence-transformers langchain_nvidia_ai_endpoints

# Expose the port the app runs on
EXPOSE 7000

# Run the Flask app
CMD ["python", "app.py"]
