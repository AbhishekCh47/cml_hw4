
# MNIST on Kubernetes (HW4)

This project demonstrates an end-to-end machine learning workflow using Kubernetes on Google Cloud Platform (GCP). It trains a model on the MNIST dataset, saves it to persistent storage, and serves inferences via a Flask web app.

## Project Structure

```
.
├── config_yaml/         # Kubernetes YAMLs for PVC, training job, inference deployment, and service
├── inference/           # Flask-based inference app and Dockerfile
├── model_train/         # PyTorch training script and Dockerfile
└── ssb10002_hw4.pdf     # Project report
```

## Setup Instructions

1. **Build and push Docker images**:
   ```bash
   docker build -t model_train ./model_train
   docker push hyperion101010/model_train:latest

   docker build -t inference ./inference
   docker push hyperion101010/inference:latest
   ```

2. **Provision resources on GKE**:
   ```bash
   kubectl apply -f config_yaml/pvc.yaml
   kubectl apply -f config_yaml/train.yaml
   kubectl apply -f config_yaml/inference.yaml
   kubectl apply -f config_yaml/lb.yaml
   ```

3. **Access the app**:
   - Get the external IP from the service:
     ```bash
     kubectl get svc service-pod-name
     ```
   - Open the IP in a browser and upload a digit image via the form.

## Key Features

- Training job runs as a Kubernetes Job and saves weights to a PersistentVolume.
- Flask app loads model weights from the same PVC for inference.
- Liveness/readiness probes ensure service health.
- Fully containerized and orchestrated on GCP.

---

© Shivam Balikondwar — NYU CS, 2025
