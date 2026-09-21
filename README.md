Travel & Tourism — Machine Learning & MLOps Capstone

An end-to-end Travel & Tourism machine learning project covering flight price prediction, REST API serving, Docker containerization, Kubernetes deployment, MLflow experiment tracking, user gender classification, and hotel recommendation through a Streamlit application.

The project is developed in Google Colab using three datasets:

flights.csv

users.csv

hotels.csv

Project Overview

The goal of this project is to demonstrate how machine learning can be used for multiple travel-related use cases and how a trained model can be moved beyond a notebook into a deployable application.

The project contains three main machine learning tasks:

1. Flight Price Prediction

A regression model predicts flight prices using route, flight type, agency, duration, distance, and date-derived features.

2. Gender Classification

A classification model predicts user gender from age and company information after removing unknown gender records.

3. Hotel Recommendation

A clustering-based recommendation system groups hotels according to hotel name, destination, stay duration, and price and uses those clusters to recommend hotels according to user preferences.

Datasets

Flights Dataset

The flights.csv dataset contains 271,888 records and 10 columns in the notebook.

Column

Description

travelCode

Travel identifier

userCode

User identifier

from

Flight origin

to

Flight destination

flightType

Flight type/class

price

Flight price and regression target

time

Flight duration

distance

Flight distance

agency

Flight agency

date

Flight date

The notebook reports no missing values in the flight dataset.

Users Dataset

The notebook loads users.csv with the following fields:

Column

Description

code

User identifier

company

User's company

name

User name

gender

Gender

age

Age

The notebook reports 1,340 user records.

Gender distribution before filtering:

Male: 452

Female: 448

None: 440

Records with gender = none are removed before classification, leaving 900 records.

Hotels Dataset

The notebook reports 40,552 hotel records across 8 columns.

Important fields used for recommendation include:

Column

Description

travelCode

Travel identifier

userCode

User identifier

name

Hotel name

place

Hotel destination

days

Number of days

price

Price per day

total

Total stay price

date

Booking date

The notebook reports no missing values in the hotel dataset.

Project Architecture

                  ┌──────────────────────┐
                  │   Travel Datasets    │
                  │ Flights / Users /    │
                  │ Hotels               │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Data Exploration &   │
                  │ Preprocessing        │
                  └──────────┬───────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ▼                ▼                ▼
   ┌────────────────┐ ┌───────────────┐ ┌────────────────┐
   │ Flight Price   │ │ Gender        │ │ Hotel          │
   │ Regression     │ │ Classification│ │ Recommendation │
   │ Random Forest  │ │ Random Forest │ │ K-Means        │
   └───────┬────────┘ └───────────────┘ └───────┬────────┘
           │                                    │
           ▼                                    ▼
   ┌────────────────┐                    ┌───────────────┐
   │ MLflow         │                    │ Streamlit     │
   │ Tracking       │                    │ Application   │
   └───────┬────────┘                    └───────────────┘
           │
           ▼
   ┌────────────────┐
   │ Flask REST API │
   └───────┬────────┘
           │
           ▼
   ┌────────────────┐
   │ Docker         │
   └───────┬────────┘
           │
           ▼
   ┌────────────────┐
   │ Kubernetes     │
   │ 3 Replicas     │
   └────────────────┘

1. Flight Price Prediction

Objective

Predict the price of a flight using historical flight data.

Data Preprocessing

The notebook performs the following preprocessing steps:

Categorical Encoding

One-hot encoding is applied to:

from

to

flightType

agency

using:

pd.get_dummies(..., drop_first=True)

Date Processing

The original date column is converted to datetime.

Four additional features are extracted:

year

month

day

dayofweek

Feature Selection

The following columns are removed from the model input:

price
travelCode
userCode
date

The target variable is:

price

Train/Test Split

The dataset is divided using:

train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

This produces:

80% Training Data
20% Testing Data

Regression Model

The project uses:

RandomForestRegressor

with:

random_state=42

Evaluation Metrics

The model is evaluated using:

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

R-squared (R²)

The notebook also generates an Actual vs. Predicted Flight Prices scatter plot with a perfect prediction reference line.

The notebook's reported baseline shows approximately:

MAE = 0.00
MSE = 0.00
R²  = 1.00

Such performance is unusually high for a real-world prediction problem. The notebook itself identifies potential data leakage or dataset characteristics as reasons to investigate the result further. Cross-validation and evaluation on a genuinely unseen dataset are recommended before treating the model as production-ready.

2. Model Persistence

The trained Random Forest regression model is saved using joblib:

random_forest_model.pkl

This file is later loaded by the Flask API and Docker application.

3. Flask REST API

The trained regression model is exposed through a Flask REST API.

Endpoint

POST /predict

The endpoint:

Receives JSON data.

Converts the request into a pandas DataFrame.

Passes the input to the saved Random Forest model.

Returns the predicted flight price as JSON.

Example response:

{
  "prediction": 1434.38
}

API Port

The Flask application runs on:

5000

The notebook starts Flask in a separate thread inside Google Colab.

For external access from Colab, the notebook notes that a tunneling service such as ngrok may be required.

Important implementation note

The current Flask/Docker implementation assumes that the incoming API data already matches the feature structure expected by the trained model.

The notebook itself notes that a production implementation should move the same preprocessing/feature-engineering logic into the API or save the feature schema alongside the model.

4. Docker Containerization

The notebook creates:

requirements.txt
Dockerfile
app.py
random_forest_model.pkl

requirements.txt

The generated requirements file contains:

Flask
joblib
pandas
scikit-learn

Docker Image

The Dockerfile uses:

python:3.9-slim-buster

The container:

Creates /app as the working directory.

Installs Python dependencies.

Copies the trained model.

Copies app.py.

Exposes port 5000.

Starts the Flask application.

Build

docker build -t flight-price-predictor .

Run

docker run -p 5000:5000 flight-price-predictor

The notebook reports that Docker execution was not available in its environment, so the image must be built and run on a Docker-enabled machine, VM, or cloud environment.

5. Kubernetes Deployment

The project prepares Kubernetes configuration for scalable deployment.

Deployment

The notebook's Kubernetes design specifies:

Deployment
Replicas: 3
Container Port: 5000
Application: flight-price-predictor

Three replicas allow multiple instances of the prediction service to run inside the cluster.

Service

The notebook creates:

service.yaml

with:

Service Type: NodePort
Port: 5000
Target Port: 5000

Apply Deployment

kubectl apply -f deployment.yaml

Apply Service

kubectl apply -f service.yaml

Verify

kubectl get deployments
kubectl get pods
kubectl get services

Kubernetes provides the deployment framework for running multiple instances of the model-serving application and managing them inside a cluster.

6. MLflow Experiment Tracking

MLflow is integrated into the regression training workflow.

What is tracked?

The notebook logs:

Parameters

random_state = 42

Metrics

MAE

MSE

R²

Model

The Random Forest model is logged using:

mlflow.sklearn.log_model(...)

The notebook registers the model as:

RandomForestFlightPricePredictor

MLflow UI

The notebook launches MLflow on:

0.0.0.0:5001

Command:

mlflow ui --host 0.0.0.0 --port 5001

When using Google Colab or another remote environment, a tunneling mechanism may be required to access the UI from a browser.

7. Gender Classification

Objective

Predict the user's gender from user-level information.

Data Preparation

The notebook:

Removes rows where gender is none.

Converts:

female → 0

male → 1

Drops:

code

name

One-hot encodes company.

Uses the remaining data as the classification feature set.

The resulting features are primarily:

age
company-related one-hot encoded features

Classification Model

The project uses:

RandomForestClassifier

with:

random_state=42

The data is split using an 80/20 train-test split.

The notebook reports:

Training samples: 720
Testing samples: 180

Classification Metrics

The model is evaluated using:

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

The notebook reports:

Accuracy  = 0.47
Precision = 0.55
Recall    = 0.41
F1 Score  = 0.47

The confusion matrix is also visualized with Seaborn.

The notebook concludes that the current features are weak predictors for this task and recommends additional feature engineering or different information sources.

8. Hotel Recommendation System

Objective

Build a clustering-based hotel recommendation system using hotel attributes.

Features

The notebook selects:

name
place
days
price

Categorical Features

The following are one-hot encoded:

name
place

Numerical Features

The following are standardized using StandardScaler:

days
price

Recommendation Model

The notebook uses:

KMeans

with:

n_clusters = 5
random_state = 42

Each hotel receives a:

cluster_label

The notebook reports that Cluster 3 contains the largest group, with 12,618 hotel records.

9. Model Artifacts for Recommendation

The notebook saves the following files:

kmeans_model.pkl
scaler.pkl
hotel_features_with_clusters.csv

These artifacts are later consumed by the Streamlit application.

10. Streamlit Web Application

The project creates:

streamlit_app.py

The application provides a user-facing hotel recommendation interface.

User Inputs

The sidebar allows the user to select:

Number of Days

Desired Price per Night

Preferred Hotel Name

Preferred Destination

The application:

Receives the user's preferences.

Applies one-hot encoding.

Aligns the input with the training feature columns.

Scales the numerical fields.

Predicts the user's K-Means cluster.

Filters hotels from that cluster.

Applies hotel/destination preferences.

Displays up to five unique recommendations.

Run the application

streamlit run streamlit_app.py

Streamlit Recommendation Flow

User Preferences
      ↓
Input Preprocessing
      ↓
One-Hot Encoding
      ↓
Numerical Scaling
      ↓
K-Means Cluster Prediction
      ↓
Filter Hotels
      ↓
Apply Name/Destination Preferences
      ↓
Display Recommendations

Current Limitations and Future Improvements

The notebook identifies several areas that can be improved.

Flight Prediction

The nearly perfect regression metrics should be investigated for:

data leakage

target-related information

dataset-specific relationships

generalization to unseen data

Recommended improvements:

cross-validation

new unseen test data

stronger preprocessing pipeline

explicit model feature schema

model monitoring

Flask API

The current API expects the input structure to match the trained model features.

A stronger production implementation should:

perform preprocessing inside the API

validate input fields

store and load the feature schema

return clearer validation errors

add health checks and logging

Gender Classification

The current model has low predictive performance:

Accuracy = 0.47
F1 = 0.47

Possible improvements include:

additional relevant features

feature engineering

alternative algorithms

stronger validation

The notebook specifically suggests investigating whether information from the dropped name field could provide useful signals, although using such information would require careful consideration of privacy, fairness, and responsible use.

Hotel Recommendation

The current recommendation system is based on K-Means similarity rather than user interaction history.

Recommended improvements:

ratings

booking history

click behavior

collaborative filtering

hybrid recommendation

personalized ranking

The Streamlit implementation also relies on row/index alignment between the processed hotel data and the original hotel dataset. A unique hotel identifier would make this linkage more robust.

Project Files

The notebook generates or uses the following major project files:

.
├── flights.csv
├── users.csv
├── hotels.csv
│
├── random_forest_model.pkl
├── requirements.txt
├── app.py
├── Dockerfile
│
├── deployment.yaml
├── service.yaml
│
├── kmeans_model.pkl
├── scaler.pkl
├── hotel_features_with_clusters.csv
│
└── streamlit_app.py

Technology Stack

Component

Technology

Programming Language

Python

Notebook Environment

Google Colab

Data Processing

Pandas

Numerical Computing

NumPy

Machine Learning

Scikit-learn

Regression

Random Forest Regressor

Classification

Random Forest Classifier

Recommendation

K-Means

Visualization

Matplotlib, Seaborn

API

Flask

Model Persistence

Joblib

Experiment Tracking

MLflow

Containerization

Docker

Deployment

Kubernetes

Web Application

Streamlit

Google Colab Execution Flow

Run the notebook in this order:

1. Load flights.csv
        ↓
2. Explore and preprocess flight data
        ↓
3. Feature engineering
        ↓
4. Train/test split
        ↓
5. Train RandomForestRegressor
        ↓
6. Evaluate regression model
        ↓
7. Save random_forest_model.pkl
        ↓
8. Create Flask API
        ↓
9. Generate Docker files
        ↓
10. Prepare Kubernetes YAML
        ↓
11. Install and configure MLflow
        ↓
12. Track regression experiment
        ↓
13. Train gender classifier
        ↓
14. Evaluate classification model
        ↓
15. Load hotel and user datasets
        ↓
16. Build K-Means recommendation model
        ↓
17. Save recommendation artifacts
        ↓
18. Build Streamlit application

MLOps Scope in the Current Notebook

The notebook directly implements or generates:

Implemented

Model training

Model evaluation

Flask REST API

Docker configuration

Kubernetes Deployment/Service configuration

MLflow experiment/model tracking

Streamlit recommendation application

Model artifact persistence

Not Fully Implemented in the Notebook

Apache Airflow: The notebook refers to integrating MLflow with a model-training/evaluation DAG, but it does not provide a complete runnable Airflow DAG in the uploaded notebook.

Jenkins: The uploaded notebook does not contain a Jenkinsfile or a complete Jenkins CI/CD pipeline.

These can be added as the next MLOps layer when moving the project from the Colab implementation to a full CI/CD production workflow.

How to Run the Main Components

Regression Notebook

Open the notebook in Google Colab and execute the cells sequentially.

Flask

python app.py

API:

http://localhost:5000/predict

Docker

docker build -t flight-price-predictor .
docker run -p 5000:5000 flight-price-predictor

Kubernetes

kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

MLflow

mlflow ui --host 0.0.0.0 --port 5001

Streamlit

streamlit run streamlit_app.py

Key Learning Outcomes

This project demonstrates the transition from a traditional notebook-based ML workflow to an application-oriented machine learning system.

The overall lifecycle is:

Data
  ↓
Exploration
  ↓
Preprocessing
  ↓
Feature Engineering
  ↓
Model Training
  ↓
Model Evaluation
  ↓
Model Persistence
  ↓
Experiment Tracking
  ↓
REST API
  ↓
Docker
  ↓
Kubernetes
  ↓
User-Facing Application

The project therefore demonstrates both machine learning development and the foundations of production-oriented ML engineering/MLOps.

Future MLOps Extensions

To complete the remaining production workflow, the project can be extended with:

GitHub
   ↓
Jenkins CI/CD
   ↓
Docker Image Build
   ↓
Container Registry
   ↓
Kubernetes Deployment

and:

Apache Airflow
      ↓
Scheduled Data Processing
      ↓
Model Training
      ↓
Evaluation
      ↓
MLflow Tracking
      ↓
Model Registry
      ↓
Deployment

Additional improvements can include automated testing, model monitoring, data drift detection, automated retraining, API authentication, centralized logging, and cloud deployment.

Conclusion

This Travel & Tourism capstone combines three machine learning use cases with model serving and deployment technologies.

The project covers:

Flight price prediction

Gender classification

Hotel recommendation

Flask REST API

Docker containerization

Kubernetes deployment

MLflow tracking

Streamlit application

The notebook provides the core implementation for the complete machine learning workflow and the deployment foundation, while Airflow and Jenkins remain natural extensions for completing the end-to-end automated MLOps lifecycle.
