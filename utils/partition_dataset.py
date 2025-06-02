import os
import argparse
import shutil
from pathlib import Path
import random
import logging
import json
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_partition(source_dir: Path, dest_dir: Path, file_list: list, class_dirs: list) -> dict:
    """
    Create a partition with all images in a single folder and generate label mappings
    
    Args:
        source_dir: Source directory containing the dataset
        dest_dir: Destination directory for the partition
        file_list: List of files to include in this partition
        class_dirs: List of class directories
        
    Returns:
        dict: Statistics about the partition and image label mappings
    """
    # Create images directory
    images_dir = dest_dir / "images"
    os.makedirs(images_dir, exist_ok=True)
    
    # Initialize statistics and label mappings
    stats = {
        "total_images": len(file_list),
        "classes": {class_dir: 0 for class_dir in class_dirs},
        "class_distribution": {}
    }
    
    image_labels = {}
    
    # Copy files and create label mappings
    for file_path in tqdm(file_list, desc=f"Creating {dest_dir.name}"):
        class_name = file_path.parent.name
        
        # Create unique filename to avoid conflicts
        unique_filename = f"{class_name}_{file_path.name}"
        dest_path = images_dir / unique_filename
        
        # Copy file and record label
        shutil.copy2(file_path, dest_path)
        image_labels[unique_filename] = class_name
        
        # Update statistics
        stats["classes"][class_name] += 1
    
    # Calculate class distribution
    total_files = len(file_list)
    stats["class_distribution"] = {
        class_name: count / total_files 
        for class_name, count in stats["classes"].items()
    }
    
    # Save label mappings
    with open(dest_dir / "image_labels.json", "w") as f:
        json.dump(image_labels, f, indent=4)
    
    # Save statistics
    with open(dest_dir / "partition_stats.json", "w") as f:
        json.dump(stats, f, indent=4)
    
    return stats

def partition_dataset(source_dir: str, output_dir: str, num_partitions: int = int(os.getenv('NUM_PARTITIONS', 4))) -> None:
    """
    Partition the dataset into multiple training parts
    
    Args:
        source_dir: Directory containing the cleaned dataset
        output_dir: Directory to save the partitioned dataset
        num_partitions: Number of partitions to create
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Get all class directories
    class_dirs = [d.name for d in source_path.iterdir() if d.is_dir()]
    logger.info(f"Found classes: {class_dirs}")
    
    # Collect all image files per class
    class_files = {}
    for class_dir in class_dirs:
        class_path = source_path / class_dir
        files = list(class_path.glob("*.[jJ][pP][gG]")) + list(class_path.glob("*.[pP][nN][gG]"))
        class_files[class_dir] = files
        logger.info(f"Found {len(files)} images for class {class_dir}")
    
    # Create balanced partitions
    partition_stats = {}
    for i in range(num_partitions):
        partition_files = []
        for class_dir, files in class_files.items():
            # Calculate files per class per partition
            files_per_class = len(files) // num_partitions
            start_idx = i * files_per_class
            end_idx = (i + 1) * files_per_class if i < num_partitions - 1 else len(files)
            partition_files.extend(files[start_idx:end_idx])
        
        # Shuffle files for this partition
        random.shuffle(partition_files)
        
        # Create partition directory
        partition_dir = output_path / f"partition_{i+1}"
        
        # Create partition with new format
        stats = create_partition(source_path, partition_dir, partition_files, class_dirs)
        
        # Save partition metadata
        partition_stats[f"partition_{i+1}"] = stats
        
        logger.info(f"Created partition_{i+1} with {len(partition_files)} training files")
    
    # Save overall partitioning statistics
    with open(output_path / "partitioning_stats.json", "w") as f:
        json.dump({
            "num_partitions": num_partitions,
            "partitions": partition_stats
        }, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Partition a dataset into multiple training parts')
    parser.add_argument('--source_dir', type=str, required=True,
                      help='Directory containing the cleaned dataset')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                      help='Directory to save the partitioned dataset')
    parser.add_argument('--num_partitions', type=int, 
                      default=int(os.getenv('NUM_PARTITIONS', 4)),
                      help='Number of partitions to create')
    parser.add_argument('--seed', type=int, 
                      default=int(os.getenv('RANDOM_SEED', 42)),
                      help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Partition dataset
    partition_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        num_partitions=args.num_partitions
    )
    
    logger.info("""
Partitioning completed! The data is organized as follows:

processed/
├── partition_1/
│   ├── images/           # All training images
│   ├── image_labels.json # Image to label mappings
│   └── partition_stats.json
├── partition_2/
└── ...

Each partition contains:
1. Single 'images' folder with all images
2. image_labels.json mapping file
3. partition_stats.json with statistics
""")

if __name__ == "__main__":
    main() 