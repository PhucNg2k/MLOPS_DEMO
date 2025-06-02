import os
import random
import shutil
import logging
from pathlib import Path
import json
import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm
from dotenv import load_dotenv

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
logger.info("Loading environment variables from .env file...")
load_dotenv()

# Log the values (without sensitive data)
logger.info(f"Using VALIDATE_BUCKET: {os.getenv('VALIDATE_BUCKET')}")
logger.info(f"Using AWS_DEFAULT_REGION: {os.getenv('AWS_DEFAULT_REGION')}")
logger.info(f"Using VALIDATION_SPLIT: {os.getenv('VALIDATION_SPLIT', '0.1')}")
logger.info(f"Using RANDOM_SEED: {os.getenv('RANDOM_SEED', '42')}")

def initialize_s3_client():
    """
    Initialize S3 client with explicit credentials from environment variables
    """
    try:
        # Use os.environ to ensure credentials exist
        try:
            aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID']
            aws_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']
            aws_region = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
        except KeyError as e:
            raise ValueError(f"Required AWS credential {e} not found in environment variables")
        
        s3 = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region
        )
        
        # Test the connection
        s3.list_buckets()
        return s3
        
    except ClientError as e:
        logger.error(f"Failed to initialize S3 client: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error initializing S3 client: {str(e)}")
        raise

def upload_to_s3(s3_client, file_path: str, bucket: str, s3_key: str):
    """
    Upload a file to S3 with error handling
    """
    try:
        logger.info(f"Uploading {file_path} to s3://{bucket}/{s3_key}")
        s3_client.upload_file(file_path, bucket, s3_key)
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_message = e.response.get('Error', {}).get('Message', str(e))
        logger.error(f"S3 upload failed - {error_code}: {error_message}")
        logger.error(f"Bucket: {bucket}")
        logger.error(f"Key: {s3_key}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during upload: {str(e)}")
        raise

def create_and_upload_validation_set(source_dir: str, 
                                   val_size: float = float(os.getenv('VALIDATION_SPLIT', 0.1)), 
                                   seed: int = int(os.getenv('RANDOM_SEED', 42)),
                                   bucket_path: str = os.getenv('VALIDATE_BUCKET')) -> dict:
    """
    Create validation set and upload to S3. All images will be in a single folder
    with a JSON file mapping image names to their labels.
    
    Args:
        source_dir: Directory containing the cleaned dataset
        val_size: Fraction of data to use for validation
        seed: Random seed for reproducibility
        bucket_path: Full S3 bucket path (bucket/prefix), can include s3:// prefix
    """
    if not bucket_path:
        raise ValueError("VALIDATE_BUCKET environment variable must be set")
    
    # Remove s3:// prefix if present
    bucket_path = bucket_path.replace('s3://', '')
    
    # Split bucket path into bucket name and prefix
    bucket_parts = bucket_path.split('/', 1)
    bucket = bucket_parts[0]
    prefix = bucket_parts[1] if len(bucket_parts) > 1 else ''
    
    # Initialize S3 client
    s3 = initialize_s3_client()
    
    random.seed(seed)
    source_path = Path(source_dir)
    
    # Create temporary directory for validation set
    temp_val_dir = Path("temp_validation")
    if temp_val_dir.exists():
        shutil.rmtree(temp_val_dir)
    os.makedirs(temp_val_dir)
    
    # Create images directory
    images_dir = temp_val_dir / "images"
    os.makedirs(images_dir)
    
    stats = {
        "total_images": 0,
        "classes": {},
        "class_distribution": {},
        "empty_classes": [],
        "split_ratio": val_size,
        "seed": seed,
        "s3_location": f"s3://{bucket}/{prefix}".rstrip('/')  # Ensure clean S3 URL format
    }
    
    # Dictionary to store image to label mappings
    image_labels = {}
    
    try:
        # Process each class
        logger.info("Creating validation set...")
        for class_dir in source_path.iterdir():
            if not class_dir.is_dir():
                continue
                
            class_name = class_dir.name
            class_files = list(class_dir.glob("*.[jJ][pP][gG]")) + list(class_dir.glob("*.[pP][nN][gG]"))
            
            if not class_files:
                logger.warning(f"No images found for class: {class_name}")
                stats["empty_classes"].append(class_name)
                continue
            
            # Select validation files
            num_val_files = max(1, int(len(class_files) * val_size))  # Ensure at least 1 file
            val_files = random.sample(class_files, k=num_val_files)
            
            # Copy validation files to single directory and record labels
            logger.info(f"Processing class {class_name}...")
            for file in tqdm(val_files, desc=f"Copying {class_name}"):
                # Create unique filename to avoid conflicts
                unique_filename = f"{class_name}_{file.name}"
                dest_path = images_dir / unique_filename
                shutil.copy2(file, dest_path)
                
                # Record label mapping
                image_labels[unique_filename] = class_name
            
            # Update statistics
            stats["classes"][class_name] = len(val_files)
            stats["class_distribution"][class_name] = len(val_files) / len(class_files)
            stats["total_images"] += len(val_files)
        
        if stats["total_images"] == 0:
            raise ValueError("No images found in any class!")
        
        if stats["empty_classes"]:
            logger.warning(f"Found empty classes: {', '.join(stats['empty_classes'])}")
        
        # Save label mappings
        labels_file = temp_val_dir / "image_labels.json"
        with open(labels_file, "w") as f:
            json.dump(image_labels, f, indent=4)
        
        # Save statistics
        stats_file = temp_val_dir / "validation_set_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=4)
        
        # Upload to S3
        logger.info(f"Uploading validation set to S3: {stats['s3_location']}")
        
        # Upload all files
        for root, _, files in os.walk(temp_val_dir):
            for file in tqdm(files, desc="Uploading to S3"):
                file_path = os.path.join(root, file)
                s3_key = os.path.join(prefix, os.path.relpath(file_path, temp_val_dir))
                s3_key = s3_key.replace("\\", "/")  # Fix Windows paths
                upload_to_s3(s3, file_path, bucket, s3_key)
        
        logger.info(f"""
Validation set creation and upload completed!

Statistics:
- Total validation images: {stats['total_images']}
- Classes: {len(stats['classes'])}
- Images per class:
{json.dumps(stats['classes'], indent=2)}
{"" if not stats['empty_classes'] else f"- Empty classes: {', '.join(stats['empty_classes'])}"}

Uploaded to: {stats['s3_location']}
""")
        
        return stats
        
    except Exception as e:
        logger.error(f"Error during validation set creation: {str(e)}")
        raise
        
    finally:
        # Clean up temporary directory
        if temp_val_dir.exists():
            shutil.rmtree(temp_val_dir)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Create and upload validation set to S3')
    parser.add_argument('--source_dir', type=str, default='data/clean',
                      help='Directory containing the cleaned dataset')
    parser.add_argument('--val_size', type=float, 
                      default=float(os.getenv('VALIDATION_SPLIT', 0.1)),
                      help='Fraction of data to use for validation')
    parser.add_argument('--bucket_path', type=str, 
                      default=os.getenv('VALIDATE_BUCKET'),
                      help='Full S3 bucket path (bucket/prefix)')
    parser.add_argument('--seed', type=int, 
                      default=int(os.getenv('RANDOM_SEED', 42)),
                      help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    try:
        # Create and upload validation set
        stats = create_and_upload_validation_set(
            source_dir=args.source_dir,
            val_size=args.val_size,
            seed=args.seed,
            bucket_path=args.bucket_path
        )
    except Exception as e:
        logger.error(f"Failed to create and upload validation set: {str(e)}")
        raise

if __name__ == "__main__":
    main() 