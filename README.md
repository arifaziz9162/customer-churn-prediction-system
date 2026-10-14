# Customer Churn Prediction System | End-to-End MLOps Pipeline

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![BentoML](https://img.shields.io/badge/BentoML-Model%20Serving-FFB800)
![Docker](https://img.shields.io/badge/Docker-Containerization-2496ED?logo=docker)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Deployment-326CE5?logo=kubernetes)
![GitHub Actions](https://img.shields.io/badge/GitHub-Actions-2088FF?logo=github-actions)
![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-13ADC7)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2)
![Airflow](https://img.shields.io/badge/Airflow-Orchestration-017CEE?logo=apacheairflow)
![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-E6522C?logo=prometheus)
![Grafana](https://img.shields.io/badge/Grafana-Dashboards-F46800?logo=grafana)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-336791?logo=postgresql)
![AWS](https://img.shields.io/badge/AWS-ECR%20%26%20EC2-FF9900?logo=amazonaws)
![License](https://img.shields.io/badge/License-MIT-green)

---

# Project Overview

Customer Churn Prediction is an end-to-end Machine Learning and MLOps project that predicts whether a customer is likely to discontinue a service based on their demographic information, account details, service subscriptions, and billing behavior.

The project demonstrates a production-oriented machine learning workflow by integrating automated data ingestion, data validation, preprocessing, model training, experiment tracking, model versioning, model evaluation, model serving, observability, containerization, CI/CD automation, and cloud deployment.

The solution follows a modular architecture that separates the machine learning pipeline, inference pipeline, model serving layer, monitoring infrastructure, and deployment workflows, making the system scalable, maintainable, and easier to extend.

The primary objective of this project is not only to build an effective customer churn prediction model but also to demonstrate modern MLOps practices used in real-world machine learning applications. By identifying customers who are likely to leave a service, the system can help businesses understand churn risk and support data-driven customer retention strategies.

---

# Key Features

- End-to-End Machine Learning Pipeline
- Modular Project Structure
- Automated Data Ingestion
- Automated Data Validation
- Automated Data Transformation
- Automated Model Training
- Automated Model Evaluation
- Hyperparameter Optimization Using Optuna
- Automated Model Registry with Champion-vs-Challenger Promotion
- Automated Model Packaging into the BentoML Model Store
- Workflow Orchestration Using Apache Airflow
- BentoML Model Serving
- Prometheus Metrics and Alerting
- Grafana Dashboards for Real-Time Observability
- Kubernetes Deployment Manifests
- PostgreSQL Database Integration
- Database Schema Management Using Alembic
- Versioned Database Migrations
- Dockerized Application
- Development, Production, and Airflow Docker Compose Environments
- Data Versioning Using DVC
- Experiment Tracking Using MLflow
- Remote Experiment Tracking Using DagsHub
- GitHub Actions Continuous Integration
- Docker Image Build & Publish to Amazon ECR
- Automated Deployment to Amazon EC2
- Automated Security Scanning
- Automated GitHub Release

---

# Dataset

This project uses the Telco-Customer-Churn dataset from Kaggle.

**Source:**
Kaggle: `WA_Fn-UseC_-Telco-Customer-Churn`

The dataset is downloaded programmatically using the Kaggle API during the data ingestion stage.

The downloaded dataset is then tracked and versioned using DVC.

---

# Technology Stack

## Programming Language

- Python 3.11

---

## Machine Learning

- Scikit-learn
- NumPy
- Pandas
- joblib
- Optuna

---

## Model Serving

- BentoML
- Pydantic

---

## Monitoring

- Prometheus
- Grafana

---

## Container Orchestration

- Kubernetes
- Kustomize

---

## Database

- PostgreSQL
- SQLAlchemy
- Psycopg2

---

## Database Migrations

- Alembic

---

## MLOps

- DVC
- MLflow
- DagsHub
- Optuna

---

## Orchestration

- Apache Airflow

---

## DevOps

- Docker
- Docker Compose
- GitHub Actions

---

## Cloud

- Amazon Elastic Container Registry (Amazon ECR)
- Amazon Elastic Compute Cloud (Amazon EC2)

---

## Development Tools

- Git
- GitHub
- Conda
- Visual Studio Code

---

# Project Structure

```text
customer-churn-prediction-system/
├── .dvcignore
├── .github
│   ├── dependabot.yml
│   └── workflows
│       ├── ci.yml
│       ├── deploy.yml
│       ├── docker.yml
│       ├── model-training.yml
│       ├── release.yml
│       └── security.yml
├── .gitignore
├── LICENSE
├── Makefile
├── README.md
├── airflow
│   ├── Dockerfile
│   ├── README.md
│   └── dags
│       └── customer_churn_dag.py
├── alembic.ini
├── bentofile.yaml
├── config
│   ├── config.yaml
│   ├── params.yaml
│   └── schema.yaml
├── deployment
│   ├── docker
│   │   └── Dockerfile.migrate
│   ├── k8s
│   │   └── base
│   │       ├── bento-deployment.yaml
│   │       ├── grafana.yaml
│   │       ├── kustomization.yaml
│   │       ├── migrate-job.yaml
│   │       ├── namespace.yaml
│   │       ├── postgres.yaml
│   │       ├── prometheus.yaml
│   │       └── secret.env.example
│   └── monitoring
│       ├── grafana
│       │   ├── dashboards
│       │   │   └── wine-quality-overview.json
│       │   └── provisioning
│       │       ├── dashboards
│       │       └── datasources
│       └── prometheus
│           ├── alert_rules.yml
│           ├── prometheus.compose.yml
│           └── prometheus.k8s.yml
├── docker-compose.airflow.yml
├── docker-compose.dev.yml
├── docker-compose.prod.yml
├── docs
│   └── images
├── dvc.lock
├── dvc.yaml
├── main.py
├── migrations
│   ├── README
│   ├── env.py
│   ├── script.py.mako
│   └── versions
│       ├── 07da342dafc0_create_churn_data_table.py
│       └── f1fd6cb99ede_create_prediction_history_table.py
├── notebooks
│   ├── EDA.ipynb
│   └── Experiments.ipynb
├── pyproject.toml
├── requirements-dev.txt
├── requirements.txt
├── serving
│   ├── __init__.py
│   ├── config.py
│   ├── constants.py
│   ├── metrics
│   │   ├── __init__.py
│   │   ├── feature_metrics.py
│   │   ├── model_metrics.py
│   │   └── prediction_metrics.py
│   ├── persistence
│   │   ├── __init__.py
│   │   └── prediction_repository.py
│   ├── schemas
│   │   ├── __init__.py
│   │   ├── model_info_schema.py
│   │   ├── predictions_schema.py
│   │   ├── request_schema.py
│   │   └── response_schema.py
│   ├── service.py
│   └── services
│       ├── __init__.py
│       ├── model_metadata.py
│       ├── prediction_service.py
│       └── quality_label.py
├── src
│   └── customer_churn_prediction
│       ├── __init__.py
│       ├── components
│       │   ├── __init__.py
│       │   ├── data_ingestion.py
│       │   ├── data_transformation.py
│       │   ├── data_validation.py
│       │   ├── model_evaluation.py
│       │   ├── model_packaging.py
│       │   ├── model_registry.py
│       │   └── model_trainer.py
│       ├── config
│       │   ├── __init__.py
│       │   └── configuration.py
│       ├── constants
│       │   └── __init__.py
│       ├── database
│       │   ├── __init__.py
│       │   ├── columns.py
│       │   ├── connection.py
│       │   └── operations.py
│       ├── entity
│       │   ├── __init__.py
│       │   └── config_entity.py
│       ├── logger
│       │   ├── __init__.py
│       │   ├── exception.py
│       │   └── logger_config.py
│       ├── pipeline
│       │   ├── __init__.py
│       │   ├── stage_01_data_ingestion.py
│       │   ├── stage_02_data_validation.py
│       │   ├── stage_03_data_transformation.py
│       │   ├── stage_04_model_trainer.py
│       │   ├── stage_05_model_evaluation.py
│       │   ├── stage_06_model_registry.py
│       │   └── stage_07_model_packaging.py
│       └── utils
│           ├── __init__.py
│           └── common.py
├── template.py
└── tests
    ├── integration
    │   └── test_pipeline_e2e.py
    ├── serving
    │   ├── test_prediction_repository.py
    │   ├── test_prediction_service.py
    │   ├── test_quality_label.py
    │   └── test_schemas.py
    └── unit
        ├── test_data_ingestion.py
        ├── test_data_transformation.py
        ├── test_data_validation.py
        ├── test_model_evaluation.py
        ├── test_model_packaging.py
        ├── test_model_registry.py
        └── test_model_trainer.py
```

> **Note**
>
> The `artifacts/` directory is generated automatically after running the machine learning pipeline. It contains the
trained model, evaluation metrics, status files, and other pipeline outputs. Most files under `artifacts/` are versioned
through DVC rather than Git; a small set of `status.json` and `metrics.json` files are explicitly force-added to Git so
that pipeline status and results remain visible directly in the repository history.

> **Note**
>
> `deployment/k8s/base/secret.env` is never committed to the repository — only `secret.env.example` is. Copy the
example file, fill in real values locally, and keep `secret.env` out of version control.

---

# Installation & Setup

## Prerequisites

Make sure the following software is installed before running the project.

- Python 3.11
- Git
- Docker
- Docker Compose
- PostgreSQL
- DVC
- Conda (Miniconda or Anaconda)
- kubectl and kustomize — required only for Kubernetes deployment
- Kaggle account with API credentials (`kaggle.json`) — required for data ingestion
- AWS CLI configured with valid credentials — required for deployment (optional for local development)

---

## Clone the Repository

```bash
git clone https://github.com/arifaziz9162/customer-churn-prediction-system.git

cd customer-churn-prediction-system
```

---

## Create Conda Environment

## Create Conda Environments

This project uses two separate Conda environments to isolate development and production dependencies. The development environment is used for data processing, model training, experimentation, and testing, while the production environment is used to run the model-serving application.

### Development Environment

```bash
conda create -n churn_dev python=3.11 -y
conda activate churn_dev
```

---

## Install Dependencies

### Development Environment

Upgrade pip.

```bash
python -m pip install --upgrade pip
```

Install the required packages.

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

Install the project as a package.

```bash
python -m pip install -e .
```

### Production Environment

Create a separate environment for the production application.

```bash
conda create -n churn_prod python=3.11 -y
conda activate churn_prod
```

Upgrade pip.

```bash
python -m pip install --upgrade pip
```

Install the production dependencies.

```bash
python -m pip install -r serving/requirements.txt
```

Install the project package without reinstalling its dependencies.

```bash
python -m pip install -e . --no-deps
```

---

## Verify the Installation

Run these commands in each environment.

```bash
python --version
python -m pip check
```

---

## Environment Variables

Create a local `.env` file in the project root for environment-specific settings and credentials.

Keep sensitive information, such as database passwords and DagsHub tokens, out of version control. Use `.env.example` as a template if available.

---

## Activate an Environment

Activate the development environment for model training and experimentation.

```bash
conda activate churn_dev
```

Activate the production environment for local serving tasks.

```bash
conda activate churn_prod
```

---

## Configure Environment Variables

Create a `.env` file in the project root directory.

```env
ENV=development

DB_HOST=localhost
DB_PORT=5432
DB_NAME=wine_quality_db
DB_USER=your_username
DB_PASSWORD=your_password

KAGGLE_USERNAME=your_kaggle_username
KAGGLE_API_TOKEN=your_kaggle_api_token

MLFLOW_TRACKING_URI=https://dagshub.com/TechArif/customer-churn-prediction-system.mlflow
DAGSHUB_USERNAME=your_username
DAGSHUB_TOKEN=your_dagshub_token

# BentoML serving
BENTO_MODEL_TAG=customer_churn_model:kmcgjpwfiwitgaav

# Application configuration
APP_VERSION=1.0.0
REQUEST_TIMEOUT_SECONDS=30
MAX_CONCURRENCY=32

# Prediction persistence
DB_SAVE_ENABLED=true
DB_SAVE_WORKERS=2

# Grafana
GRAFANA_ADMIN_PASSWORD=your_grafana_password

# Airflow's isolated metadata database (used only by docker-compose.airflow.yml)
AIRFLOW_DB_USER=your_db_username
AIRFLOW_DB_PASSWORD=your_db_password
AIRFLOW_DB_NAME=your_db_name
```

---

## Configure PostgreSQL

Create the PostgreSQL database:

```sql
CREATE DATABASE customer_churn_db;
```

Update the database credentials inside the `.env` file before running the application.

The database tables are created and updated through Alembic migrations.

---

## Database Migrations

The project uses Alembic to manage and version the PostgreSQL database schema. Alembic has already been initialized
for this project (see the `migrations/` directory) — running `alembic init` again is not required.

Create a Migration

```bash
alembic revision -m "describe the schema change"
```

View Migration History

```bash
alembic history
```

Apply Migrations

```bash
alembic upgrade head
```

Roll back the latest Migration

```bash
alembic downgrade -1
```

Roll back all Migrations

```bash
alembic downgrade base
```

---

# Running the Project

## Run the Machine Learning Pipeline

Execute the complete machine learning pipeline (training environment).

```bash
python main.py
```

The pipeline performs the following stages:

- Data Ingestion
- Data Validation
- Data Transformation
- Model Training
- Model Evaluation
- Model Registry (champion-vs-challenger promotion)

Model Packaging (stage 7) packages the current Production model into the BentoML model store and is run separately,
from the serving environment — see below.

---

## Hyperparameter Optimization

Optuna is used for hyperparameter optimization of the selected machine learning model.

The optimization process searches for better hyperparameter combinations using cross-validation and the configured
scoring metric.

Typical workflow:
Model Experiments
↓
Model Comparison
↓
Select Final Model
↓
Optuna Optimization
↓
Final Hyperparameters
↓
Model Evaluation


Optuna configuration is maintained in `params.yaml`.

---

## Package and Run the BentoML Service

From the **serving environment**, package the current Production model into the BentoML model store.

```bash
python -m wine_quality_prediction.pipeline.stage_07_model_packaging
```

Build the Bento.

```bash
bentoml build -f bentofile.yaml
```

Serve it locally.

```bash
bentoml serve serving.service:WineQualityService --reload
```

The service will be available at:

```text
http://localhost:3000
```

Interactive API Documentation (Swagger):

```text
http://localhost:3000/docs
```

---

## Running with Docker

### Development Environment

Build the BentoML image once.

```bash
bentoml containerize wine_quality_service:latest -t customer-churn-service:latest
```

Build and start the development containers (PostgreSQL, Alembic migration, BentoML service, Prometheus, Grafana).

```bash
docker compose -f docker-compose.dev.yml up --build -d
```

Stop the development containers.

```bash
docker compose -f docker-compose.dev.yml down
```

### Production Environment

Build and start the production containers.

```bash
docker compose -f docker-compose.prod.yml up --build -d
```

Stop the production containers.

```bash
docker compose -f docker-compose.prod.yml down
```

### Airflow Orchestration

Build and start the Airflow environment (scheduler, webserver, and its own metadata database).

```bash
docker compose -f docker-compose.airflow.yml up -d
```
Airflow UI:

```text
http://localhost:8080
```

Stop the Airflow environment.

```bash
docker compose -f docker-compose.airflow.yml down
```

See `airflow/README.md` for more details on how the pipeline is orchestrated through DVC. Note that Model Packaging
(stage 7) intentionally does not run inside Airflow — it needs `bentoml`, which cannot be installed alongside
`dagshub` in the same container. It only runs through GitHub Actions, in its own job.

---

## Running on Kubernetes

Copy the secrets template and fill in real values.

```bash
cp deployment/k8s/base/secret.env.example deployment/k8s/base/secret.env
```

Push the BentoML image to your container registry, then update the image reference in
`deployment/k8s/base/bento-deployment.yaml`.

Apply the manifests.

```bash
kubectl apply -k deployment/k8s/base --load-restrictor LoadRestrictionsNone
```

Wait for the database migration Job to finish.

```bash
kubectl wait --for=condition=complete job/wine-quality-migrate -n wine-quality --timeout=120s
```

Check that everything is running.

```bash
kubectl get pods -n wine-quality
```

Access the service, Prometheus, and Grafana locally.

```bash
kubectl port-forward -n customer-churn svc/customer-churn-service 3000:3000
kubectl port-forward -n customer-churn svc/prometheus 9090:9090
kubectl port-forward -n customer-churn svc/grafana 3001:3000
```

Tear everything down.

```bash
kubectl delete -k deployment/k8s/base --load-restrictor LoadRestrictionsNone
```

---

## Monitoring

Prometheus:

```text
http://localhost:9090
```

Grafana (default login: `admin` / the password set in `GRAFANA_ADMIN_PASSWORD`):

```text
http://localhost:3001
```

The Grafana dashboard and Prometheus alert rules are auto-provisioned — no manual setup is required after the stack
starts. Tracked signals include request latency, error rate, replicas per model version, served model R2, prediction
distribution, and per-feature input drift.

---

## DVC

This project uses DVC (Data Version Control) to manage datasets, track pipeline dependencies, and reproduce machine learning workflows. DVC works alongside Git to help maintain reproducibility across development environments.

### Pull Data and Artifacts

Download tracked data and artifacts from the configured DVC remote.

```bash
dvc pull
```

### Check Pipeline Status

Check whether the data or pipeline outputs differ from the versions recorded by DVC.

```bash
dvc status
```

### Reproduce the Pipeline

Execute the pipeline stages defined in `dvc.yaml`. DVC reruns stages whose dependencies or parameters have changed.

```bash
dvc repro
```

### Push Data and Artifacts

Upload tracked data and artifacts to the configured DVC remote.

```bash
dvc push
```

### Visualize the Pipeline

Generate a diagram showing the dependencies between pipeline stages.

```bash
dvc dag
```

---

## MLflow Experiment Tracking

Start the MLflow tracking server.

```bash
mlflow ui
```

Open the MLflow dashboard.

```text
http://localhost:5000
```

Training runs, parameters, metrics, and artifacts are automatically logged during model training and evaluation.

---

## DagsHub Integration

This project uses DagsHub as the remote MLflow tracking server.

Configure the following environment variables.

```env
DAGSHUB_USERNAME=your_username
DAGSHUB_TOKEN=your_token

MLFLOW_TRACKING_URI=https://dagshub.com/TechArif/customer-churn-prediction-system.mlflow
```

After configuration, all experiment metadata, parameters, metrics, and model artifacts are automatically synchronized
with the DagsHub repository.

---

## Verify Installations

Verify that all required tools are installed correctly.

```bash
python --version

pip --version

alembic --version

docker --version

docker compose --version

dvc --version

mlflow --version

bentoml --version

kubectl version --client
```

If all commands execute successfully, the project is configured correctly.

---

# API Endpoints

The BentoML service exposes REST APIs for prediction, health, and monitoring.

| Method | Endpoint | Description |
|--------|-----------|-------------|
| GET | `/docs` | Swagger UI |
| GET | `/livez` | Liveness Probe |
| GET | `/readyz` | Readiness Probe |
| GET | `/metrics` | Prometheus Metrics |
| POST | `/api/v1/predict` | Predict Wine Quality |
| POST | `/api/v1/model_info` | Served Model Version and Metrics |
| POST | `/api/v1/predictions` | Fetch Saved Predictions |

---

# CI/CD Pipeline

The project uses GitHub Actions to automate testing, containerization, migrations, and deployment.

## Dependency Management

- Automated Python dependency updates
- Automated GitHub Actions updates
- Automated Docker base image updates

## Continuous Integration

- Code quality checks (Ruff, Black) for both the training package and the serving package
- Project structure validation
- Training-side unit and integration testing
- Serving-side testing, in its own job/environment
- Test coverage reporting

## Security

- CodeQL analysis
- Trivy vulnerability scanning
- Gitleaks secret scanning

## Docker

- Package the current Production model into the BentoML model store
- Build and push the BentoML service image to Amazon ECR
- Build and push the database migration image to Amazon ECR

## Model Training Pipeline

- Pull dataset and artifacts using DVC
- Train model
- Optimize model hyperparameters with Optuna
- Evaluate model
- Validate model performance using quality gates
- Register the model with champion-vs-challenger promotion
- Upload model artifacts and evaluation reports

## Database

- PostgreSQL database
- Database schema managed using Alembic
- Apply pending Alembic migrations during deployment, via a dedicated migration image
- Keep PostgreSQL schema versioned
- Maintain reproducible database schema changes

## Deployment

- Deploy Docker containers to Amazon EC2
- Authenticate with Amazon ECR
- Pull Docker images from Amazon ECR
- Stop existing application containers
- Start the PostgreSQL container
- Apply pending Alembic migrations
- Start the BentoML service container
- Start Prometheus and Grafana containers
- Verify running containers
- Perform application health checks

## Release

- Automatic GitHub Release creation
- Package model artifacts, reports, and DVC metadata
- Upload release artifact

---

# Machine Learning Pipeline

The training pipeline consists of seven stages.

1. Data Ingestion
2. Data Validation
3. Data Transformation
4. Model Training
5. Model Evaluation
6. Model Registry (champion-vs-challenger promotion)
7. Model Packaging (MLflow Production → BentoML model store)

Hyperparameter optimization with Optuna is performed as part of the model training stage.

---

# MLOps Workflow

- Data versioning using DVC
- Experiment tracking using MLflow
- Remote experiment tracking with DagsHub
- Hyperparameter optimization using Optuna
- Continuous Training with automatic champion-vs-challenger model promotion
- Model packaging and serving using BentoML
- Database schema versioning using Alembic
- PostgreSQL database integration
- Workflow orchestration using Apache Airflow
- Observability using Prometheus and Grafana
- Docker containerization
- Kubernetes deployment manifests
- CI/CD using GitHub Actions
- Container registry using Amazon ECR
- Deployment on Amazon EC2

---

# Screenshots

## MLflow Dashboard

![MLflow Dashboard](docs/images/mlflow.png)

## DagsHub Experiments

![DagsHub Experiments](docs/images/dagshub.png)

## GitHub Actions (CI/CD)

![GitHub Actions](docs/images/github-actions.png)

## Amazon ECR Repositories

![Amazon ECR Repositories](docs/images/ecr.png)

## Docker Containers

![Docker Containers](docs/images/docker.png)

## BentoML Service

![BentoML Service](docs/images/bentoml.png)

## Kubernetes Deployment

![Kubernetes Deployment](docs/images/kubernetes.png)

## Kubernetes Pods

![Kubernetes Pods](docs/images/kubernetes-pods.png)

## Prometheus Monitoring

![Prometheus Monitoring](docs/images/prometheus.png)

## Grafana Dashboard

![Grafana Dashboard](docs/images/grafana.png)

## Amazon EC2 Deployment

![AWS EC2 Deployment](docs/images/ec2.png)

---

# Performance

Metrics from the final trained XGBoost model (see `artifacts/model_evaluation/metrics.json` for the current run).

| Metric | Value |
|--------|-------|
| Cross-Validation Recall | 0.8174 |
| Train Recall | 0.9492 |
| Test Recall | 0.9305 |
| Train-Test Recall Gap | 0.0187 |
| Train Accuracy | 0.6179 |
| Test Accuracy| 0.6118 |
| Train Precision | 0.4059 |
| Test Precision | 0.4005 |
| Train F1 | 0.5686 |
| Test F1 | 0.5599 |
| Train ROC-AUC | 0.8568 |
| Test ROC-AUC | 0.8410 |
| Train PR-AUC | 0.6736 |
| Test PR-AUC | 0.6426 |
| Classification Threshold | 0.30 |
| Estimated Test Cost | 39,050.00 |

---

## Model Artifacts

The training and deployment pipelines generate the following artifacts:

### Data Pipeline

* **Raw Dataset (`artifacts/data_ingestion/raw_data.csv`):** Stores the ingested customer churn dataset.
* **Training Dataset (`artifacts/data_transformation/train.csv`):** Contains the transformed training data.
* **Test Dataset (`artifacts/data_transformation/test.csv`):** Contains the held-out test data.
* **Validation Report (`artifacts/data_validation/validation_report.json`):** Records data validation results.

### Model Training and Evaluation

* **Trained Model (`artifacts/model_trainer/model.joblib`):** Serialized machine learning pipeline used to generate churn predictions.
* **Training Summary (`artifacts/model_trainer/training_summary.json`):** Records the MLflow run ID, model URI, selected hyperparameters, classification threshold, cross-validation score, and training metrics.
* **Threshold Analysis (`artifacts/model_trainer/threshold_sweep.csv`):** Contains evaluation metrics and estimated business costs for different classification thresholds.
* **Evaluation Metrics (`artifacts/model_evaluation/metrics.json`):** Stores test-set classification metrics, including accuracy, precision, recall, F1-score, ROC-AUC, PR-AUC, and estimated total cost.

### Model Packaging and Registry

* **Packaging Report (`artifacts/model_packaging/packaging_report.json`):** Records the results and relevant details of the model packaging stage.
* **Registry Report (`artifacts/model_registry/registry_report.json`):** Records model registration details and registry status.

### Pipeline Status

Each pipeline stage generates a `status.json` file indicating whether the stage completed successfully.

**Note:** MLflow also tracks experiment runs and model information separately from these local artifacts. BentoML packaging and MLflow model registration are distinct operations.

---

# Repository

Clone the repository.

```bash
git clone https://github.com/arifaziz9162/customer-churn-prediction-system.git

cd customer-churn-prediction-system
```

---

# Author

**MD ARIF AZIZ**

AI/ML Engineer

GitHub: https://github.com/arifaziz9162

Email: arifaziz0125@gmail.com

---

# License

This project is licensed under the **MIT License**.

See the `LICENSE` file for more information.
