# Productionization-of-ML-Systems

This repository contains the complete end-to-end implementation of various Machine Learning tasks, their management via MLflow, containerization using Docker, orchestration instructions for Kubernetes, and an interactive frontend using Streamlit.

Project Components
1. Flight Price Prediction (Regression)
Task: Predict flight ticket prices based on historical flight records.
Pipeline:
Preprocessing with one-hot encoding on categorical attributes (from, to, flightType, agency).
Temporal feature extraction (year, month, day, dayofweek) from flight dates.
Training a RandomForestRegressor with scikit-learn.
Performance: High evaluation accuracy on the test partition.
MLflow Integration: Every model run, hyperparameter configuration, metric (MAE, MSE, $R^2$$R^2$), and final artifact model is registered using MLflow Tracking.
2. Model Serving API (Flask)
Endpoint: /predict (POST)
Implementation: A Flask web server (app.py) loads the pre-trained Random Forest model and yields predictions from incoming JSON payloads containing flight feature configurations.
3. Containerization and Orchestration
Docker: Package the Flask web service and model within a lightweight python:3.9-slim-buster container via Dockerfile.
Kubernetes: Scalable orchestration manifests (deployment.yaml, service.yaml) running 3 replicas of the server exposed through a NodePort service.
4. User Gender Classification
Task: Classify user gender ('female' / 'male') based on age and company.
Model: RandomForestClassifier trained on filtered non-null gender metadata.
5. Hotel Recommendation System (Streamlit App)
Algorithm: K-Means clustering ($K=5$$K=5$) to segment hotels depending on pricing and length of stay.
App Features: streamlit_app.py allows interactive querying of recommendations with sidebar inputs mapped directly to the nearest cluster centroids.
Quick Start Guide
Requirements
Install python packages from the auto-generated requirements:

pip install -r requirements.txt
Run Streamlit App
Launch the local interactive recommendation dashboard:

streamlit run streamlit_app.py
Run MLflow UI
Verify your experiment metrics visually:

mlflow ui --port 5001
