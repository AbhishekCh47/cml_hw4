# Use a lightweight Python 3.10 base image
FROM python:3.10-slim

# Image maintainer information
LABEL maintainer="Abhishek Chigurupati <abhishekchigurupati47@gmail.com>"

# Set the container's working directory
WORKDIR /train_service

# Copy dependency and training files into the container
COPY requirements.txt .
COPY train_model.py .

# Install Python dependencies without caching to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Run the training script with specified CLI arguments
CMD ["python3", "train_model.py", "--batch-size", "256", "--epochs", "1", "--save-model"]
