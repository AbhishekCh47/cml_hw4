# Use a minimal Python 3.10 image
FROM python:3.10-slim

# Metadata for image maintainership
LABEL maintainer="Abhishek Chigurupati <ac11950@nyu.edu>"

# Set the container's working directory
WORKDIR /app

# Copy application source and dependencies
COPY serve_model.py .
COPY requirements.txt .
COPY templates/ templates/

# Install Python dependencies without caching to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Make the Flask app accessible on port 5000
EXPOSE 5000

# Run the Flask application
CMD ["python", "serve_model.py"]
