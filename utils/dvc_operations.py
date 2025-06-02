import os
import subprocess
import logging
import shutil
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_dvc(force: bool = False) -> None:
    """
    Initialize DVC with local data storage and S3 remote
    
    Args:
        force: Whether to force reinitialize DVC
    """
    dvc_dir = Path('.dvc')
    if dvc_dir.exists() and not force:
        logger.info("DVC already initialized")
        return
        
    if force and dvc_dir.exists():
        logger.info("Forcing DVC reinitialization")
        subprocess.run(['rm', '-rf', '.dvc'], check=True)
    
    # Initialize DVC
    logger.info("Initializing DVC...")
    subprocess.run(['dvc', 'init', '--no-scm'], check=True)
    
    # Configure local data directory
    dvc_store = os.path.join(os.environ.get('PROJECT_DATA_DIR', ''), 'dvc_store')
    os.makedirs(dvc_store, exist_ok=True)
    logger.info(f"Configuring local data directory: {dvc_store}")
    subprocess.run(['dvc', 'config', 'cache.dir', dvc_store], check=True)
    
    # Configure S3 remote
    dvc_remote = os.getenv('DVC_REMOTE_URL')
    if not dvc_remote:
        raise ValueError("DVC_REMOTE_URL environment variable must be set")
    
    logger.info(f"Configuring DVC remote: {dvc_remote}")
    subprocess.run(['dvc', 'remote', 'add', '-d', 'default', dvc_remote], check=True)
    
    # Note: AWS credentials will be read from environment variables automatically
    # No need to configure them in DVC as they will be picked up from:
    # AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_DEFAULT_REGION

def track_partition(partition_path: str) -> None:
    """
    Track a new partition with DVC, storing data locally
    
    Args:
        partition_path: Path to the partition directory
    """
    partition_dir = Path(partition_path)
    if not partition_dir.exists():
        raise ValueError(f"Partition directory {partition_path} does not exist")
    
    # Ensure DVC is initialized
    init_dvc()
    
    # Add partition to DVC (data stored locally)
    logger.info(f"Adding partition to DVC: {partition_path}")
    subprocess.run(['dvc', 'add', partition_path, '-v'], check=True)
    
    # Commit DVC changes
    logger.info("Committing DVC changes...")
    subprocess.run(['dvc', 'commit', '-v'], check=True)
    
    # Push only hashes to remote storage
    logger.info("Pushing hashes to remote storage...")
    subprocess.run(['dvc', 'push', '-v'], check=True)
    
    # Create metadata file
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'partition': os.path.basename(partition_path),
        'dvc_remote': os.getenv('DVC_REMOTE_URL'),
        'dvc_remote_name': os.getenv('DVC_REMOTE_NAME', 'dvc-store'),
        'local_data_dir': os.path.join(os.environ.get('PROJECT_DATA_DIR', ''), 'dvc_store')
    }
    
    metadata_file = partition_dir / 'dvc_metadata.json'
    with open(metadata_file, 'w') as f:
        import json
        json.dump(metadata, f, indent=4)
    
    logger.info(f"Partition {partition_path} successfully tracked with DVC")

def pull_partition(partition_path: str) -> None:
    """
    Ensure partition data is available locally
    
    Args:
        partition_path: Path to the partition directory
    """
    # Ensure DVC is initialized
    init_dvc()
    
    # Check if data exists locally
    if not Path(partition_path).exists():
        logger.info(f"Partition {partition_path} not found locally, checking DVC tracking...")
        subprocess.run(['dvc', 'checkout', partition_path, '-v'], check=True)

def verify_remote() -> bool:
    """
    Verify DVC remote configuration and access
    
    Returns:
        bool: True if remote is properly configured and accessible
    """
    try:
        # Check if DVC is initialized
        if not Path('.dvc').exists():
            logger.error("DVC not initialized")
            return False
            
        # List remotes and check configuration
        result = subprocess.run(['dvc', 'remote', 'list'], 
                              capture_output=True, text=True, check=True)
        
        if 'default' not in result.stdout:
            logger.error("Default remote not found in DVC configuration")
            return False
            
        # Verify remote configuration by checking status
        try:
            subprocess.run(['dvc', 'status', '-r', 'default'], 
                         capture_output=True, check=True)
            logger.info("Successfully verified access to remote")
            return True
        except subprocess.CalledProcessError as e:
            if "Unable to authenticate" in str(e.stderr):
                logger.error("Authentication failed for remote storage")
            else:
                logger.error(f"Failed to access remote: {e.stderr.decode().strip()}")
            return False
        
    except subprocess.CalledProcessError as e:
        logger.error(f"DVC remote verification failed: {e.stdout}\n{e.stderr}")
        return False

def cleanup_cache(older_than_days: int = 30) -> None:
    """
    Clean up old data from local cache
    
    Args:
        older_than_days: Remove data older than this many days
    """
    try:
        # Clean up local cache
        subprocess.run(['dvc', 'gc', '--workspace', 
                       f'--older-than', f'{older_than_days}d'], check=True)
        
        # Push any remaining hashes
        subprocess.run(['dvc', 'push', '-v'], check=True)
        
        logger.info(f"Successfully cleaned up cache older than {older_than_days} days")
    except subprocess.CalledProcessError as e:
        logger.error(f"Cache cleanup failed: {str(e)}")
        raise

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='DVC operations for dataset versioning')
    parser.add_argument('--operation', type=str, required=True,
                      choices=['init', 'track', 'pull', 'verify', 'cleanup'],
                      help='Operation to perform')
    parser.add_argument('--partition_dir', type=str,
                      help='Path to partition directory (for track/pull operations)')
    parser.add_argument('--force', action='store_true',
                      help='Force DVC reinitialization')
    parser.add_argument('--older-than-days', type=int, default=30,
                      help='Clean up data older than this many days')
    
    args = parser.parse_args()
    
    try:
        if args.operation == 'init':
            init_dvc(args.force)
        elif args.operation == 'track':
            if not args.partition_dir:
                raise ValueError("partition_dir is required for track operation")
            track_partition(args.partition_dir)
        elif args.operation == 'pull':
            if not args.partition_dir:
                raise ValueError("partition_dir is required for pull operation")
            pull_partition(args.partition_dir)
        elif args.operation == 'verify':
            if verify_remote():
                logger.info("DVC remote configuration verified successfully")
            else:
                logger.error("DVC remote verification failed")
                exit(1)
        elif args.operation == 'cleanup':
            cleanup_cache(args.older_than_days)
            
    except Exception as e:
        logger.error(f"Operation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 