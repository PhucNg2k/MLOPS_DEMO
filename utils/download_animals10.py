import os
import shutil
import logging
from pathlib import Path
import kagglehub
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Class mapping for cleaner directory names
CLASS_MAPPING = {
    'cane': 'dog',
    'cavallo': 'horse',
    'elefante': 'elephant',
    'farfalla': 'butterfly',
    'gallina': 'chicken',
    'gatto': 'cat',
    'mucca': 'cow',
    'pecora': 'sheep',
    'scoiattolo': 'squirrel',
    'ragno': 'spider'
}

def download_animals10(output_dir: str) -> str:
    """
    Download Animals-10 dataset using kagglehub
    
    Args:
        output_dir: Directory to save the downloaded dataset
        
    Returns:
        str: Path to the extracted dataset directory
    """
    # Create a temporary directory for the download
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info("Downloading Animals-10 dataset...")
        # Download to temp directory first
        os.environ['KAGGLE_DOWNLOAD_PATH'] = temp_dir
        dataset_path = kagglehub.dataset_download(
            "alessiocorrado99/animals10"
        )
        
        # Create raw data directory in the project
        raw_dir = os.path.join(output_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        
        # Move files from temp directory to project directory
        logger.info(f"Moving dataset to project directory: {raw_dir}")
        for item in os.listdir(dataset_path):
            src = os.path.join(dataset_path, item)
            dst = os.path.join(raw_dir, item)
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            shutil.move(src, dst)
        
        logger.info(f"Dataset downloaded to: {raw_dir}")
        return raw_dir

def clean_dataset(raw_dir: str, clean_dir: str) -> str:
    """
    Clean the dataset by renaming classes to English names
    
    Args:
        raw_dir: Directory containing the raw dataset
        clean_dir: Directory to save the cleaned dataset
        
    Returns:
        str: Path to the cleaned dataset directory
    """
    raw_path = Path(os.path.join(raw_dir, "raw-img"))
    clean_path = Path(clean_dir)
    
    # Remove existing clean directory if it exists
    if clean_path.exists():
        shutil.rmtree(clean_path)
    
    logger.info("Cleaning and organizing dataset...")
    for italian_name, english_name in CLASS_MAPPING.items():
        src_dir = raw_path / italian_name
        dst_dir = clean_path / english_name
        
        if src_dir.exists():
            os.makedirs(dst_dir, exist_ok=True)
            logger.info(f"Processing class: {italian_name} -> {english_name}")
            for img_file in src_dir.glob("*"):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    shutil.copy2(img_file, dst_dir / img_file.name)
    
    logger.info("Dataset cleaned and organized")
    return str(clean_path)

def main():
    # Set up directories
    base_dir = "data"
    os.makedirs(base_dir, exist_ok=True)
    clean_dir = os.path.join(base_dir, "clean")
    
    # Download dataset
    raw_dir = download_animals10(base_dir)
    
    # Clean and organize dataset
    clean_dataset_dir = clean_dataset(raw_dir, clean_dir)
    
    logger.info(f"""
Dataset download and preparation completed!

Dataset Information:
- 10 animal classes: dog, horse, elephant, butterfly, chicken, cat, cow, sheep, squirrel, spider
- ~28,000 images total
- Images are of varying sizes and quality (realistic scenario)

Data Locations:
- Raw data: {os.path.join(raw_dir, "raw-img")}
- Cleaned data: {clean_dataset_dir}

To partition the dataset, use the partition_dataset.py script.
""")

if __name__ == "__main__":
    main() 