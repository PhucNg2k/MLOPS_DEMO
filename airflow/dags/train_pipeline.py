from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Param
import os
import json
import shutil
import subprocess
import mlflow
from airflow.utils.dates import days_ago
from dotenv import load_dotenv
import logging
import boto3
from airflow.hooks.S3_hook import S3Hook
from typing import List, Set
from datetime import datetime, timedelta
from airflow.operators.bash import BashOperator
from airflow.exceptions import AirflowFailException

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Add project paths
AIRFLOW_HOME = os.environ.get('AIRFLOW_HOME', '/opt/airflow')
PROJECT_ROOT = os.path.dirname(AIRFLOW_HOME)

# Initialize AWS connection
s3_hook = S3Hook(aws_conn_id='aws_default')

# Verify required environment variables
required_env_vars = [
    'TRAIN_BUCKET', 
    'VALIDATE_BUCKET', 
    'DVC_REMOTE_URL',
    'GITHUB_TOKEN',
    'GIT_USER_EMAIL',
    'GIT_USER_NAME',
    'GIT_REPO_URL',
    'GIT_BRANCH'
]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

logger.info(f"Using TRAIN_BUCKET: {os.getenv('TRAIN_BUCKET')}")
logger.info(f"Using VALIDATE_BUCKET: {os.getenv('VALIDATE_BUCKET')}")
logger.info(f"Using DVC_REMOTE_URL: {os.getenv('DVC_REMOTE_URL')}")

# Set MLflow tracking URI from environment variable (if present)
mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')
if mlflow_tracking_uri:
    mlflow.set_tracking_uri(mlflow_tracking_uri)

def get_processed_partitions():
    """Get list of already processed partition folders"""
    processed_path = os.path.join(AIRFLOW_HOME, 'data', 'processed_partitions.json')
    if os.path.exists(processed_path):
        try:
            with open(processed_path, 'r') as f:
                return set(json.load(f))
        except Exception as e:
            logger.error(f"Error reading processed partitions file: {str(e)}")
            return set()
    return set()

def save_processed_partition(partition):
    """Save partition as processed"""
    processed = get_processed_partitions()
    processed.add(partition)
    processed_path = os.path.join(AIRFLOW_HOME, 'data', 'processed_partitions.json')
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        
        # Write file directly without explicit permissions
        with open(processed_path, 'w') as f:
            json.dump(list(processed), f)
            
        logger.info(f"Successfully saved processed partition: {partition}")
    except Exception as e:
        logger.error(f"Error saving processed partition: {str(e)}")
        raise

def extract_partition_from_key(s3_key: str) -> str:
    """Extract partition folder name from S3 key."""
    # Example key: train/partition_1/image_labels.json
    parts = s3_key.strip('/').split('/')
    for i, part in enumerate(parts):
        if part.startswith('partition_'):
            return part
    raise ValueError(f"No partition folder found in key: {s3_key}")

# Default arguments for DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1
}

def download_partition(partition_path: str) -> str:
    """
    Download a partition from S3 to local storage
    Args:
        partition_path: Name of partition folder (e.g. 'partition_1')
    Returns:
        str: Path to the downloaded partition
    """
    if not partition_path:
        raise ValueError("partition_path cannot be None")
    
    logger.info(f"Processing partition: {partition_path}")
    
    # Create local directory for partition
    local_path = os.path.join(AIRFLOW_HOME, 'data', 'training_data', partition_path)
    
    try:
        # Check if partition already exists locally and validate its structure
        if os.path.exists(local_path):
            # Verify the structure of existing data
            required_files = ['images', 'image_labels.json', 'partition_stats.json']
            missing_files = [f for f in required_files if not os.path.exists(os.path.join(local_path, f))]
            
            if not missing_files:
                logger.info(f"Partition already exists locally at {local_path} with all required files")
                return local_path
            else:
                logger.warning(f"Found incomplete local data. Missing: {missing_files}")
                shutil.rmtree(local_path)
                logger.info("Removed incomplete local data")
        
        # Create directory for download
        os.makedirs(local_path, exist_ok=True)
        
        # Construct S3 path
        s3_path = f"{os.getenv('TRAIN_BUCKET')}/{partition_path}"
        logger.info(f"Checking S3 partition structure: {s3_path}")
        
        # Use Airflow's S3 hook
        s3 = s3_hook.get_conn()
        
        # Parse bucket and prefix
        bucket_name = s3_path.split('/')[2]
        prefix = '/'.join(s3_path.split('/')[3:])
        
        # List all objects in partition
        response = s3.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix
        )
        
        if 'Contents' not in response:
            raise ValueError(f"No files found in S3 path: {s3_path}")
            
        # Check required structure
        found_items = {
            'images': False,
            'image_labels.json': False,
            'partition_stats.json': False
        }
        
        for obj in response['Contents']:
            key = obj['Key'].replace(prefix + '/', '')
            if key.startswith('images/'):
                found_items['images'] = True
            elif key == 'image_labels.json':
                found_items['image_labels.json'] = True
            elif key == 'partition_stats.json':
                found_items['partition_stats.json'] = True
                
        # Check if any required items are missing
        missing_items = [item for item, found in found_items.items() if not found]
        if missing_items:
            raise ValueError(f"Invalid partition structure in S3. Missing: {', '.join(missing_items)}")
            
        logger.info("S3 partition structure validated successfully")
            
        # Get AWS credentials from Airflow connection
        aws_creds = s3_hook.get_credentials()
        env = os.environ.copy()
        env['AWS_ACCESS_KEY_ID'] = aws_creds.access_key
        env['AWS_SECRET_ACCESS_KEY'] = aws_creds.secret_key
        if aws_creds.token:
            env['AWS_SESSION_TOKEN'] = aws_creds.token
            
        # Download files using aws cli
        result = subprocess.run([
            'aws', 's3', 'cp', s3_path, local_path, '--recursive'
        ], capture_output=True, text=True, check=False, env=env)
        
        # Check if the error is only about utime
        if result.returncode != 0:
            stderr_lines = result.stderr.splitlines()
            only_utime_warnings = all(
                "unable to update the last modified time" in line 
                for line in stderr_lines 
                if not line.startswith("Completed")
            )
            
            if not only_utime_warnings:
                logger.error(f"AWS CLI error (stdout): {result.stdout}")
                logger.error(f"AWS CLI error (stderr): {result.stderr}")
                raise RuntimeError(f"Failed to download partition: aws s3 cp exited with {result.returncode}")
            else:
                logger.warning("Ignoring utime warnings as files were downloaded successfully")
        
        # Validate downloaded data structure
        required_files = ['images', 'image_labels.json', 'partition_stats.json']
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(local_path, f))]
        if missing_files:
            raise ValueError(f"Download incomplete. Missing files: {missing_files}")
            
        # Validate that all images in image_labels.json exist
        try:
            with open(os.path.join(local_path, 'image_labels.json'), 'r') as f:
                image_labels = json.load(f)
            
            missing_images = []
            for image_name in image_labels.keys():
                image_path = os.path.join(local_path, 'images', image_name)
                if not os.path.exists(image_path):
                    missing_images.append(image_name)
            
            if missing_images:
                logger.error(f"Found {len(missing_images)} missing images:")
                for img in missing_images[:10]:  # Show first 10 missing images
                    logger.error(f"- {img}")
                if len(missing_images) > 10:
                    logger.error(f"... and {len(missing_images) - 10} more")
                raise ValueError(f"Partition has {len(missing_images)} missing images referenced in image_labels.json")
            
            logger.info(f"Successfully validated {len(image_labels)} images")
        except json.JSONDecodeError:
            raise ValueError("Failed to parse image_labels.json")
        
        logger.info(f"Successfully downloaded and validated partition at {local_path}")
        return local_path
        
    except Exception as e:
        logger.error(f"Error processing partition: {str(e)}")
        if os.path.exists(local_path):
            shutil.rmtree(local_path)
        raise

def version_partition(partition_path: str, **context) -> None:
    """
    Version the partition using DVC. Handles both new partitions and updates to existing ones.
    
    NOTE: DVC push to remote is NOT performed here. Only local DVC tracking is done.
    DVC push is handled by GitHub Actions after .dvc files are committed and pushed to the repo.
    
    Args:
        partition_path: Local path to partition data (e.g. /opt/airflow/data/training_data/partition_1)
    """
    import subprocess
    
    partition_name = os.path.basename(partition_path)
    logger.info(f"Versioning partition with DVC: {partition_name}")
    
    try:
        # Initialize DVC if not already done
        if not os.path.exists('.dvc'):
            logger.info("Initializing DVC repository")
            subprocess.run(['dvc', 'init', '--no-scm'], check=True)
            
            # Configure remote storage (for local tracking only)
            dvc_remote = os.getenv('DVC_REMOTE_URL')
            if not dvc_remote:
                raise ValueError("DVC_REMOTE_URL environment variable must be set")
                
            logger.info("Configuring DVC remote storage (local only, no push)")
            subprocess.run(['dvc', 'remote', 'add', '--local', '-d', 'storage', dvc_remote], check=True)
            
            # Configure AWS credentials from Airflow connection
            aws_creds = s3_hook.get_credentials()
            subprocess.run(['dvc', 'remote', 'modify', '--local', 'storage', 'access_key_id', aws_creds.access_key], check=True)
            subprocess.run(['dvc', 'remote', 'modify', '--local', 'storage', 'secret_access_key', aws_creds.secret_key], check=True)
            if aws_creds.token:
                subprocess.run(['dvc', 'remote', 'modify', '--local', 'storage', 'session_token', aws_creds.token], check=True)
        
        # Check if partition is already tracked
        dvc_file = f"{partition_path}.dvc"
        is_update = os.path.exists(dvc_file)
        
        if is_update:
            logger.info(f"Partition {partition_name} already tracked, updating DVC hash")
            # DVC will automatically detect changes and update the hash
        else:
            logger.info(f"New partition {partition_name}, adding to DVC tracking")
        
        # Add/update partition in DVC
        subprocess.run(['dvc', 'add', partition_path], check=True)
        logger.info(f"DVC add complete for partition: {partition_name} (no push performed)")
        # NOTE: DVC push is intentionally omitted. See project docs and GitHub Actions workflow.
        # No hash extraction or extra logic needed; hash is visible in the .dvc file and in git history after merge.
        
        # Create/update metadata file for MLflow tracking
        metadata = {
            'timestamp': context['ts'],
            'partition': partition_name,
            'is_update': is_update
        }
        
        metadata_file = os.path.join(partition_path, 'dvc_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"{'Updated' if is_update else 'Created'} DVC metadata file: {metadata_file}")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"DVC operation failed: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error during DVC versioning: {str(e)}")
        raise

def train_model(partition_path: str, **context) -> str:
    try:
        # Set up model directories - using paths within Airflow workspace
        temp_model_dir = os.path.join(AIRFLOW_HOME, 'data', 'models', 'temp')
        git_model_dir = os.path.join(AIRFLOW_HOME, 'data', 'models', 'final')
        
        # Create directories with proper permissions
        for dir_path in [temp_model_dir, git_model_dir]:
            try:
                os.makedirs(dir_path, mode=0o755, exist_ok=True)
                logger.info(f"Created directory: {dir_path}")
            except Exception as e:
                logger.error(f"Error creating directory {dir_path}: {str(e)}")
                raise

        # Generate model filename with timestamp
        model_filename = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        temp_model_path = os.path.join(temp_model_dir, model_filename)
        final_model_path = os.path.join(git_model_dir, model_filename)

        # Start MLflow run
        with mlflow.start_run(run_name=f"train_{os.path.basename(partition_path)}"):
            # Run training script with correct arguments
            train_script = os.path.join(AIRFLOW_HOME, 'src', 'train.py')
            cmd = [
                'python', train_script,
                '--data_dir', partition_path,
                '--output_dir', temp_model_dir
            ]
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                logger.info("Training script completed successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"Training subprocess failed!\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
                raise

            # Log command output to MLflow
            mlflow.log_param("training_output", result.stdout)
            if result.stderr:
                mlflow.log_param("training_errors", result.stderr)

            # Copy model to git-tracked directory using simple file read/write
            best_model_path = os.path.join(temp_model_dir, 'best_model.pth')
            if os.path.exists(best_model_path):
                try:
                    with open(best_model_path, 'rb') as src, open(final_model_path, 'wb') as dst:
                        dst.write(src.read())
                    logger.info(f"Copied model from {best_model_path} to {final_model_path}")
                except Exception as e:
                    logger.error(f"Error copying model file: {str(e)}")
                    raise
            else:
                raise FileNotFoundError(f"Expected model file not found: {best_model_path}")

            return final_model_path
            
    except Exception as e:
        logger.error(f"Error in train_model: {str(e)}")
        raise

def cleanup(**context):
    """
    Selective cleanup of temporary files while preserving important data.
    
    We keep:
    - Training data (for reproducibility)
    - .dvc files (for model version tracking)
    - Evaluation results (for analysis and MLflow)
    
    We clean:
    - Temporary validation data (can be re-downloaded)
    - Any temporary files from training
    - .pth model files (can be retrieved using DVC)
    """
    try:
        # Clean validation data as it can be re-downloaded
        validation_dir = os.path.join(AIRFLOW_HOME, 'data', 'validation_data')
        if os.path.exists(validation_dir):
            logger.info(f"Cleaning validation data directory: {validation_dir}")
            try:
                shutil.rmtree(validation_dir)
            except PermissionError:
                logger.warning(f"Could not remove validation directory: {validation_dir}")
        
        # Clean any temporary training artifacts
        temp_model_dir = os.path.join(AIRFLOW_HOME, 'data', 'models', 'temp')
        if os.path.exists(temp_model_dir):
            logger.info(f"Cleaning temporary model directory: {temp_model_dir}")
            try:
                shutil.rmtree(temp_model_dir)
            except PermissionError:
                logger.warning(f"Could not remove temp model directory: {temp_model_dir}")

        # Clean .pth files from final models directory while keeping .dvc files
        final_model_dir = os.path.join(AIRFLOW_HOME, 'data', 'models', 'final')
        if os.path.exists(final_model_dir):
            logger.info(f"Cleaning .pth files from final model directory: {final_model_dir}")
            try:
                for file in os.listdir(final_model_dir):
                    if file.endswith('.pth'):
                        file_path = os.path.join(final_model_dir, file)
                        os.remove(file_path)
                        logger.info(f"Removed model file: {file}")
            except Exception as e:
                logger.warning(f"Error while removing .pth files: {str(e)}")
        
        # List preserved directories and files for logging
        preserved_dirs = [
            os.path.join(AIRFLOW_HOME, 'data', 'training_data'),
            os.path.join(AIRFLOW_HOME, 'data', 'evaluation_results')
        ]
        
        logger.info("Preserved directories for analysis:")
        for dir_path in preserved_dirs:
            if os.path.exists(dir_path):
                logger.info(f"- {dir_path}")

        # Log preserved .dvc files
        if os.path.exists(final_model_dir):
            dvc_files = [f for f in os.listdir(final_model_dir) if f.endswith('.dvc')]
            logger.info(f"Preserved {len(dvc_files)} .dvc files for model tracking")
            
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        # Don't raise the error since cleanup is not critical
        # Just log it and continue
    return None

with DAG(
    'animal_classification_training_v16',
    default_args=default_args,
    description='Train animal classification model on new partitions',
    schedule_interval='0 */2 * * *',  # Run every 2 hours
    catchup=False
) as dag:

    # NOTE: DVC push to remote is handled by GitHub Actions, not by Airflow. Airflow only tracks data locally with DVC.
    # See project README and .github/workflows for details.

    # 1. List all partition folders in TRAIN_BUCKET
    def list_s3_partitions(**context):
        bucket_url = os.getenv('TRAIN_BUCKET')
        bucket_name = bucket_url.split('/')[2]
        prefix = '/'.join(bucket_url.split('/')[3:]).rstrip('/') + '/'
        s3 = s3_hook.get_conn()
        paginator = s3.get_paginator('list_objects_v2')
        partitions = set()
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix, Delimiter='/'):
            # Use CommonPrefixes to get folder names
            for cp in page.get('CommonPrefixes', []):
                folder = cp['Prefix'][len(prefix):].rstrip('/')
                if folder.startswith('partition_'):
                    partitions.add(folder)
            # Fallback: also check keys for legacy/flat structure
            for obj in page.get('Contents', []):
                key = obj['Key']
                rel = key[len(prefix):]
                parts = rel.split('/')
                if parts[0].startswith('partition_'):
                    partitions.add(parts[0])
        logger.info(f"[DEBUG] list_s3_partitions: Found partitions: {partitions}")
        processed = get_processed_partitions()
        unprocessed = [p for p in sorted(partitions) if p not in processed]
        logger.info(f"[DEBUG] list_s3_partitions: Unprocessed partitions: {unprocessed}")
        if not unprocessed:
            logger.info("No new unprocessed partitions found. Skipping run.")
            return None
        selected_partition = unprocessed[0]
        logger.info(f"[DEBUG] list_s3_partitions: Selected partition: {selected_partition}")
        return selected_partition

    list_partitions_task = PythonOperator(
        task_id='list_s3_partitions',
        python_callable=list_s3_partitions,
        provide_context=True
    )

    # 2. Check if partition is already processed (redundant, but keeps logic modular)
    def check_partition_not_processed(partition, **context):
        if not partition:
            logger.info("No new partition found in S3.")
            raise AirflowFailException("No new partition found to process. Failing the DAG run.")
            
        processed = get_processed_partitions()
        if partition in processed:
            logger.info(f"Partition {partition} already processed.")
            raise AirflowFailException("Selected partition was already processed. Failing the DAG run.")
            
        logger.info(f"Processing partition: {partition}")
        return partition

    check_partition_task = PythonOperator(
        task_id='check_partition_not_processed',
        python_callable=check_partition_not_processed,
        op_kwargs={
            'partition': "{{ task_instance.xcom_pull(task_ids='list_s3_partitions') }}"
        },
        provide_context=True
    )

    # 3. Download partition (now receives partition name)
    def download_partition_with_log(partition_path: str, **context):
        if not partition_path:
            logger.info("No partition to download. Skipping.")
            return None
        result = download_partition(partition_path)
        logger.info(f"[DEBUG] download_partition: Passing local path '{result}' to downstream task.")
        return result

    download_task = PythonOperator(
        task_id='download_partition',
        python_callable=download_partition_with_log,
        op_kwargs={
            'partition_path': "{{ task_instance.xcom_pull(task_ids='check_partition_not_processed') }}"
        },
        retries=2,  # Retry 2 times
        retry_delay=timedelta(minutes=1),  # Wait 1 minute between retries
        retry_exponential_backoff=True,  # Exponentially increase delay between retries
        max_retry_delay=timedelta(minutes=5)  # Cap the delay at 5 minutes
    )

    # 4. Version with DVC
    def version_partition_with_log(partition_path: str, **context):
        if not partition_path:
            logger.info("No partition to version. Skipping.")
            return None
        version_partition(partition_path, **context)
        logger.info(f"[DEBUG] version_partition: Versioned partition at '{partition_path}'. Passing to downstream task.")
        return partition_path

    version_task = PythonOperator(
        task_id='version_partition',
        python_callable=version_partition_with_log,
        op_kwargs={
            'partition_path': "{{ task_instance.xcom_pull(task_ids='download_partition') }}"
        }
    )

    # 5. Train model
    def train_model_with_log(partition_path: str, **context):
        if not partition_path:
            logger.info("No partition to train on. Skipping.")
            return None
        try:
            model_path = train_model(partition_path, **context)
            logger.info(f"[DEBUG] train_model: Passing model path '{model_path}' to downstream task.")
            return model_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed with error:\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
            raise

    train_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model_with_log,
        op_kwargs={
            'partition_path': "{{ task_instance.xcom_pull(task_ids='download_partition') }}"
        }
    )

    # Add git push task - Only push .dvc file to GitHub, actual model goes to S3
    git_push = BashOperator(
        task_id='git_push',
        bash_command=f"""
            # Create and move to a temporary directory
            TEMP_DIR=$(mktemp -d) && \
            cd $TEMP_DIR && \
            
            # Configure git
            git config --global user.email "$GIT_USER_EMAIL" && \
            git config --global user.name "$GIT_USER_NAME" && \
            
            # Clone the repository (using token)
            git clone "https://$GITHUB_TOKEN@github.com/$GIT_REPO_NAME" . && \
            
            # Get the latest model file and copy it with proper path
            latest_model=$(ls -t {AIRFLOW_HOME}/data/models/final/model_*.pth | head -n1) && \
            
            # Ensure directory exists and copy with full path structure
            mkdir -p data/models/final && \
            
            # Copy new model
            cp "$latest_model" "data/models/final/$(basename $latest_model)" && \
            
            # Initialize DVC and track the model file (no remote needed)
            dvc init --no-scm && \
            dvc add "data/models/final/$(basename $latest_model)" && \
            
            # Upload model to S3 directly using AWS CLI
            aws s3 cp "data/models/final/$(basename $latest_model)" "$DVC_REMOTE_URL/$(basename $latest_model)" && \
            
            # Remove the .pth file, keep only .dvc
            rm "data/models/final/$(basename $latest_model)" && \
            
            # Add the .dvc file to git (with -f to override gitignore)
            git add -f "data/models/final/$(basename $latest_model).dvc" && \
            
            # Add pipeline code for version tracking
            mkdir -p airflow/dags && \
            cp {AIRFLOW_HOME}/dags/train_pipeline.py airflow/dags/ && \
            git add airflow/dags/train_pipeline.py && \
            
            # Multiple safety checks for credentials
            echo "Running security checks..." && \
            
            # Check 1: No .env files
            if git diff --cached --name-only | grep -i "\.env$"; then
                echo "Error: Attempting to commit .env file"
                exit 1
            fi && \
            
            # Check 2: No DVC local configs
            if git diff --cached --name-only | grep -i "\.dvc/config.local$\|\.dvc/tmp"; then
                echo "Error: Attempting to commit DVC local config"
                exit 1
            fi && \
            
            # Check 3: No credential patterns in actual values
            if git diff --cached | grep -i -v "os.getenv\|os.environ\|secrets\.\|env\[\|get_credentials\|AWS_\|GITHUB_" | grep -i -E "(password|credential|secret|token).*[=:].+"; then
                echo "Error: Potential hardcoded credentials found"
                exit 1
            fi && \
            
            # Commit and push
            git commit -m "Add model tracking for evaluation
            
            Model: $(basename $latest_model)
            Note: Model file uploaded to S3" && \
            git remote set-url origin "https://$GITHUB_TOKEN@github.com/$GIT_REPO_NAME" && \
            git push origin $GIT_BRANCH && \
            
            # Cleanup
            cd /tmp && rm -rf $TEMP_DIR
        """,
        env={
            'GIT_USER_EMAIL': os.getenv('GIT_USER_EMAIL'),
            'GIT_USER_NAME': os.getenv('GIT_USER_NAME'),
            'GIT_BRANCH': os.getenv('GIT_BRANCH', 'dev'),
            'GITHUB_TOKEN': os.getenv('GITHUB_TOKEN'),
            'GIT_REPO_NAME': os.getenv('GIT_REPO_URL').split('github.com/')[-1].rstrip('.git'),
            'DVC_REMOTE_URL': os.getenv('DVC_REMOTE_URL'),
            'AWS_ACCESS_KEY_ID': os.getenv('AWS_ACCESS_KEY_ID'),
            'AWS_SECRET_ACCESS_KEY': os.getenv('AWS_SECRET_ACCESS_KEY'),
            'AWS_DEFAULT_REGION': os.getenv('AWS_DEFAULT_REGION', 'us-east-1'),
            'PATH': os.getenv('PATH'),  # Ensure AWS CLI and DVC are available
            'HOME': '/home/airflow'  # Set HOME for git config
        }
    )

    # 6. Cleanup (skip evaluation step)
    def cleanup_with_log(**context):
        cleanup(**context)
        logger.info(f"[DEBUG] cleanup: Cleanup complete. Passing None to downstream task.")
        return None

    cleanup_task = PythonOperator(
        task_id='cleanup',
        python_callable=cleanup_with_log
    )

    # 7. Mark partition as processed
    def _mark_processed(**context):
        partition = context['task_instance'].xcom_pull(task_ids='check_partition_not_processed')
        if partition:
            save_processed_partition(partition)
        logger.info(f"[DEBUG] mark_processed: Marked partition '{partition}' as processed.")
        return partition

    mark_processed = PythonOperator(
        task_id='mark_processed',
        python_callable=_mark_processed
    )

    # Set task dependencies (no evaluation step)
    list_partitions_task >> check_partition_task >> download_task >> version_task >> train_task >> git_push >> cleanup_task >> mark_processed