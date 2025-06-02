# Animal Classification MLOps Pipeline

This project implements a complete MLOps pipeline for training and deploying an animal classification model. The pipeline automates the entire workflow from data preparation to model deployment, featuring:

- Automated training on new data partitions using Airflow
- Model versioning with DVC and AWS S3
- Experiment tracking with MLflow
- Automated evaluation and deployment using GitHub Actions
- Quality gates to ensure model performance

## Prerequisites

- Python 3.10
- Docker Desktop with WSL2 (for Windows users)
- AWS Account with S3 bucket
- GitHub Account (for forking and Actions)

## Architecture

### Training Pipeline (Airflow)
- Monitors S3 for new training data partitions
- Automatically triggers training on new data
- Versions models and pushes to S3
- Tracks experiments with MLflow

### Evaluation & Deployment (GitHub Actions)
- Triggered by new model versions
- Downloads model from S3
- Runs evaluation on validation dataset
- Auto-merges to main if accuracy > 85%
- Creates issues for failed evaluations

### Storage & Versioning
- AWS S3: Stores models and datasets
- DVC: Tracks model versions
- GitHub: Stores code and .dvc files
- MLflow: Tracks experiments and metrics

## Initial Setup

1. **Fork and Clone the Repository**
   - Fork this repository to your GitHub account
   - Clone your forked repository:
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

2. **Initialize DVC**
```bash
# Install DVC if not already installed
pip install dvc[s3]

# Initialize DVC in the repository
dvc init
```

Note: DVC remote configuration is handled automatically by Airflow using environment variables defined in `.env`. No additional DVC setup is required.

3. **Configure GitHub Repository**
   - Go to your forked repository's Settings
   - Under "Secrets and variables" > "Actions"
   - Add the following repository secrets:
     - `AWS_ACCESS_KEY_ID`
     - `AWS_SECRET_ACCESS_KEY`
     - `AWS_DEFAULT_REGION`
     - `VALIDATE_BUCKET`
     - `DVC_REMOTE_URL`
     - `GITHUB_TOKEN`

## Setup Environment

1. **Create Environment File**

Create a `.env` file in the project root:
```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=your_region

# S3 Buckets
TRAIN_BUCKET=s3://your-bucket/train    # For training partitions
VALIDATE_BUCKET=s3://your-bucket       # For validation dataset
DVC_REMOTE_URL=s3://your-bucket/models # For model storage

# GitHub Configuration
GITHUB_TOKEN=your_github_token
GIT_USER_EMAIL=your.email@example.com
GIT_USER_NAME="Your Name"
GIT_REPO_URL=https://github.com/username/repo
GIT_BRANCH=dev

# MLflow Configuration
MLFLOW_TRACKING_URI=http://mlflow:5000
```

2. **Install Data Preparation Tools**
```bash
# Make sure you have Python 3.10 installed
python --version  # Should show Python 3.10.x

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages for data preparation utils
pip install boto3 kagglehub pillow tqdm pandas
```

## Data Preparation

1. **Download Dataset**
```bash
# Download and clean Animals-10 dataset
python utils/download_animals10.py
```

2. **Create Training Partitions**
```bash
# Split dataset into training partitions
python utils/partition_dataset.py
```

3. **Prepare Validation Set**
```bash
# Create and upload validation dataset
python utils/prepare_validation_set.py
```

4. **Upload First Training Partition**
```bash
# Upload first partition to start training
python utils/upload_partitions.py --partition_path data/processed/partition_2
```

## Docker Setup

1. **WSL2 Setup (Windows Only)**
   - Install WSL2:
     ```powershell
     # Run in PowerShell as Administrator
     wsl --install
     wsl --set-default-version 2
     ```
   - Configure Docker Desktop to use WSL2 backend
   - Clone/move project to WSL2 filesystem for better performance

2. **Build and Start Services**
```bash
# Build Docker images
docker compose build

# Initialize Airflow
docker compose run airflow-init

# Start all services
docker compose up -d
```

Access service UIs at:
- Airflow: http://localhost:8080 (airflow/airflow)
- MLflow: http://localhost:5000

## Using the Pipeline

### Adding New Training Data
1. Prepare new data partition
2. Upload to S3:
```bash
python utils/upload_partitions.py --partition_path data/training_data/partition_X
```
3. Airflow will automatically detect and train on new data
4. GitHub Actions will evaluate the new model

### Manual Model Evaluation
1. Go to GitHub Actions
2. Select "Evaluate and Deploy Model"
3. Click "Run workflow"
4. Select dev branch
5. Add reason for manual run

### Accessing Models
- Latest production model: See main branch .dvc files
- Historical models: Check dev branch history
- Download specific version:
```bash
# Switch to the desired branch (main for production models)
git checkout main

# Pull latest .dvc files
git pull origin main

# Download the model using DVC (will store in local cache first)
dvc pull data/models/final/model_YYYYMMDD_HHMMSS.pth.dvc

# The model file will be available at data/models/final/model_YYYYMMDD_HHMMSS.pth
# A copy is also kept in your local DVC cache (dvc-store) for future use
```

### Monitoring
- Training progress: MLflow UI
- Pipeline status: Airflow UI
- Model evaluations: GitHub Actions logs
- Production models: Main branch .dvc files