import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
from model import get_model
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
import logging
import boto3
from dotenv import load_dotenv
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
logger.info("Loading environment variables from .env file...")
load_dotenv()

# Log the values (without sensitive data)
logger.info(f"Using VALIDATE_BUCKET: {os.getenv('VALIDATE_BUCKET')}")
logger.info(f"Using AWS_DEFAULT_REGION: {os.getenv('AWS_DEFAULT_REGION')}")

class ValidationDataset(Dataset):
    """Custom dataset for validation data with JSON labels"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.transform = transform
        
        # Load image labels
        with open(self.data_dir / "image_labels.json") as f:
            self.image_labels = json.load(f)
            
        # Load validation stats for class mapping
        with open(self.data_dir / "validation_set_stats.json") as f:
            self.stats = json.load(f)
            
        # Create class to index mapping
        self.classes = sorted(list(self.stats["classes"].keys()))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Create list of (image_path, label_idx) pairs
        self.samples = [
            (self.images_dir / img_name, self.class_to_idx[label])
            for img_name, label in self.image_labels.items()
        ]
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def download_validation_data(data_dir: str):
    """
    Download validation data from S3 bucket
    """
    bucket_path = os.getenv('VALIDATE_BUCKET')
    if not bucket_path:
        raise ValueError("VALIDATE_BUCKET environment variable must be set")
    
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
    
    logger.info(f"Downloading validation data from s3://{bucket}/{prefix}")
    
    # Create data directory and images subdirectory
    images_dir = Path(data_dir) / "images"
    os.makedirs(images_dir, exist_ok=True)
    
    # Download validation data using explicit credentials
    s3 = boto3.client('s3',
                     aws_access_key_id=aws_access_key_id,
                     aws_secret_access_key=aws_secret_access_key,
                     region_name=aws_region)
    
    try:
        # First, download metadata files
        for filename in ["validation_set_stats.json", "image_labels.json"]:
            file_key = f"{prefix}/{filename}"
            local_path = os.path.join(data_dir, filename)
            s3.download_file(bucket, file_key, local_path)
        
        # Download all images
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/images/"):
            if 'Contents' in page:
                for obj in page['Contents']:
                    if obj['Key'].lower().endswith(('.jpg', '.jpeg', '.png')):
                        file_name = os.path.basename(obj['Key'])
                        local_file = os.path.join(images_dir, file_name)
                        s3.download_file(bucket, obj['Key'], local_file)
        
        logger.info("Validation data downloaded successfully")
        
        # Load and return validation stats
        with open(os.path.join(data_dir, "validation_set_stats.json")) as f:
            return json.load(f)
            
    except Exception as e:
        logger.error(f"Error downloading validation data: {str(e)}")
        raise

def evaluate(model_path, data_dir, output_dir, threshold=0.85):
    """
    Evaluate model on validation data and determine if it passes validation
    
    Args:
        model_path: Path to the model checkpoint
        data_dir: Directory containing validation data
        output_dir: Directory to save evaluation results
        threshold: Minimum accuracy threshold to pass validation (default: 85%)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and configuration
    logger.info(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    num_classes = checkpoint['num_classes']
    class_names = checkpoint['class_names']
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load validation dataset
    logger.info(f"Loading validation data from {data_dir}")
    val_dataset = ValidationDataset(data_dir, transform=transform)
    
    # Verify class compatibility
    if set(val_dataset.classes) != set(class_names):
        raise ValueError("Validation dataset classes do not match model classes")
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    # Load model
    model = get_model(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    logger.info("Starting evaluation...")
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # Overall accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
            
            # Save predictions for detailed metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    # Calculate overall accuracy
    accuracy = 100 * correct / total
    
    # Calculate per-class accuracy
    class_accuracies = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            class_accuracies[class_names[i]] = 100 * class_correct[i] / class_total[i]
    
    # Generate detailed classification report
    report = classification_report(all_targets, all_predictions, 
                                target_names=class_names, 
                                output_dict=True)
    
    # Save detailed metrics to CSV
    metrics_df = pd.DataFrame(report).transpose()
    metrics_df.to_csv(os.path.join(output_dir, 'detailed_metrics.csv'))
    
    # Determine if model passes validation
    validation_passed = accuracy >= threshold
    
    # Save summary metrics
    metrics = {
        'test_accuracy': round(accuracy, 2),
        'threshold': round(threshold, 2),
        'validation_passed': validation_passed,
        'per_class_accuracy': {k: round(v, 2) for k, v in class_accuracies.items()},
        'total_samples': total,
        'class_distribution': {class_names[i]: class_total[i] for i in range(num_classes)}
    }
    
    metrics_file = os.path.join(output_dir, 'eval_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Log results with clear pass/fail indication
    logger.info("\nValidation Results:")
    logger.info("=" * 50)
    logger.info(f"Overall Accuracy: {accuracy:.2f}%")
    logger.info(f"Validation Threshold: {threshold:.2f}%")
    logger.info(f"Validation Status: {'PASSED' if validation_passed else 'FAILED'}")
    logger.info("\nPer-class Accuracy:")
    for class_name, acc in class_accuracies.items():
        logger.info(f"{class_name}: {acc:.2f}%")
    logger.info("=" * 50)
    
    return metrics

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate model on validation data')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the model checkpoint')
    parser.add_argument('--data_dir', type=str, default='validation_data',
                      help='Directory to store validation data')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                      help='Directory to save evaluation results')
    parser.add_argument('--download', action='store_true',
                      help='Download validation data from S3')
    parser.add_argument('--threshold', type=float, default=85.0,
                      help='Minimum accuracy threshold to pass validation (default: 85%)')
    
    args = parser.parse_args()
    
    try:
        # Download validation data if requested
        if args.download:
            download_validation_data(args.data_dir)
        
        # Run evaluation
        metrics = evaluate(args.model_path, args.data_dir, args.output_dir, args.threshold)
        
        # Exit with status code based on validation result
        if not metrics['validation_passed']:
            logger.error("Validation failed - model did not meet accuracy threshold")
            exit(1)
            
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 