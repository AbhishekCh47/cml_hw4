# MNIST Digit Classifier on Kubernetes (GKE)

A cloud-native machine learning pipeline that demonstrates containerized training and inference 
deployed on Google Kubernetes Engine using the MNIST handwritten digit dataset.

## Live Demo

**Try it yourself:** [http://34.9.53.127]

## 📋 Project Overview

This project implements a complete ML workflow on Kubernetes:
- Containerized model training with PyTorch
- Persistent model storage using Kubernetes volumes
- Real-time inference service through a web interface
- End-to-end orchestration with Kubernetes on GCP

## Repository Structure
.
├── config_yaml/         # Kubernetes configuration manifests
│   ├── persistent-volume.yaml
│   ├── mnist-data-loader.yaml
│   ├── train-job.yaml
│   ├── inference-deployment.yaml
│   └── inference-service.yaml
├── inference/           # Inference service
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── serve_model.py
│   └── templates/
│       └── mnist.html
├── model_train/         # Model training
│   ├── Dockerfile
│   ├── requirements.txt
│   └── train_model.py
└── ac11950_hw4_report.pdf     # Project documentation

## Deployment Instructions

### 1. Build and Push Docker Images

```bash
# Build and push training image
cd model_train
docker build -t abhishekchigurupati/modeltraingcp:latest .
docker push abhishekchigurupati/modeltraingcp:latest

# Build and push inference image
cd ../inference
docker build -t abhishekchigurupati/inferencegcp:latest .
docker push abhishekchigurupati/inferencegcp:latest
2. Deploy to Kubernetes
bash# Create persistent storage
kubectl apply -f config_yaml/persistent-volume.yaml

# Deploy data preparation pod
kubectl apply -f config_yaml/mnist-data-loader.yaml

# Run training job
kubectl apply -f config_yaml/train-job.yaml

# Deploy inference service
kubectl apply -f config_yaml/inference-deployment.yaml
kubectl apply -f config_yaml/inference-service.yaml
3. Access the Application
bash# Get external IP of the inference service
kubectl get svc ac11950-inference-service
Visit the external IP in a web browser to access the handwritten digit classifier.

Key Features

Microservices Architecture: Separate containers for training and inference
Stateful Machine Learning: Shared model artifacts through Kubernetes PVC
Cloud-Native Deployment: Fully orchestrated on GKE
Scalable Design: Independent scaling of training and inference components
Interactive Web Interface: User-friendly digit classification through file upload

Author
Abhishek Chigurupati (ac11950)