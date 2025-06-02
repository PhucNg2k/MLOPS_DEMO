import os
import argparse
import logging
import boto3
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
logger.info("Loading environment variables from .env file...")
load_dotenv()

# Log the values (without sensitive data)
logger.info(f"Using TRAIN_BUCKET: {os.getenv('TRAIN_BUCKET')}")
logger.info(f"Using AWS_DEFAULT_REGION: {os.getenv('AWS_DEFAULT_REGION')}")

def upload_partition(partition_path: str, bucket_path: str = os.getenv('TRAIN_BUCKET')) -> None:
    """
    Upload a single partition to S3
    
    Args:
        partition_path: Path to the partition directory
        bucket_path: Full S3 bucket path (bucket/prefix), can include s3:// prefix
    """
    if not bucket_path:
        raise ValueError("TRAIN_BUCKET environment variable must be set or bucket_path must be provided")
    
    # Ensure AWS credentials are set
    try:
        aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID']
        aws_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']
        aws_region = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
    except KeyError as e:
        raise ValueError(f"Required AWS credential {e} not found in environment variables")
    
    # Remove s3:// prefix if present
    bucket_path = bucket_path.replace('s3://', '')
    
    # Split bucket path into bucket name and prefix
    bucket_parts = bucket_path.split('/', 1)
    bucket = bucket_parts[0]
    prefix = bucket_parts[1] if len(bucket_parts) > 1 else ''
    
    partition_dir = Path(partition_path)
    if not partition_dir.exists():
        raise ValueError(f"Partition directory {partition_path} does not exist")
    
    if not partition_dir.is_dir():
        raise ValueError(f"{partition_path} is not a directory")
        
    # Verify partition structure
    required_files = ['images', 'image_labels.json', 'partition_stats.json']
    for file in required_files:
        if not (partition_dir / file).exists():
            raise ValueError(f"Partition missing required file/directory: {file}")
    
    # Initialize S3 client with explicit credentials
    try:
        s3 = boto3.client('s3',
                         aws_access_key_id=aws_access_key_id,
                         aws_secret_access_key=aws_secret_access_key,
                         region_name=aws_region)
        # Test connection by listing buckets
        s3.list_buckets()
    except Exception as e:
        logger.error(f"Failed to initialize S3 client: {str(e)}")
        raise
    
    # Get partition name from path
    partition_name = partition_dir.name
    if not partition_name.startswith('partition_'):
        raise ValueError(f"Invalid partition directory name: {partition_name}. Must start with 'partition_'")
    
    logger.info(f"Uploading partition {partition_name}...")
    
    try:
        # Upload all files in the partition
        for root, _, files in os.walk(partition_dir):
            for file in tqdm(files, desc=f"Uploading {os.path.basename(root)}"):
                file_path = os.path.join(root, file)
                s3_key = os.path.join(prefix, partition_name, 
                                    os.path.relpath(file_path, partition_dir))
                s3_key = s3_key.replace("\\", "/")  # Fix Windows paths
                
                try:
                    s3.upload_file(file_path, bucket, s3_key)
                except Exception as e:
                    logger.error(f"Failed to upload {file_path}: {str(e)}")
                    raise
                    
    except Exception as e:
        logger.error(f"Error uploading partition {partition_name}: {str(e)}")
        raise
    
    logger.info(f"Successfully uploaded partition {partition_name} to s3://{bucket}/{prefix}/{partition_name}")

def main():
    parser = argparse.ArgumentParser(description='Upload a single partition to S3')
    parser.add_argument('--partition_path', type=str, required=True,
                      help='Path to the partition directory')
    parser.add_argument('--bucket_path', type=str,
                      default=os.getenv('TRAIN_BUCKET'),
                      help='Full S3 bucket path (bucket/prefix)')
    
    args = parser.parse_args()
    
    try:
        upload_partition(
            partition_path=args.partition_path,
            bucket_path=args.bucket_path
        )
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main() 