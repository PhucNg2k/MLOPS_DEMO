import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import json
from model import get_model
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
from PIL import Image

def validate_data_directory(data_dir):
    """Validate the data directory structure and contents before training."""
    images_dir = os.path.join(data_dir, 'images')
    labels_json = os.path.join(data_dir, 'image_labels.json')
    
    # Check directory existence
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory does not exist: {data_dir}")
    if not os.path.exists(images_dir):
        raise ValueError(f"Images directory does not exist: {images_dir}")
    if not os.path.exists(labels_json):
        raise ValueError(f"Labels file does not exist: {labels_json}")
        
    # Check directory permissions
    if not os.access(data_dir, os.R_OK):
        raise PermissionError(f"Cannot read data directory: {data_dir}")
    if not os.access(images_dir, os.R_OK):
        raise PermissionError(f"Cannot read images directory: {images_dir}")
    if not os.access(labels_json, os.R_OK):
        raise PermissionError(f"Cannot read labels file: {labels_json}")
    
    # Validate labels file
    try:
        with open(labels_json, 'r') as f:
            labels = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in labels file: {str(e)}")
    
    if not isinstance(labels, dict):
        raise ValueError("Labels file must contain a dictionary")
    if not labels:
        raise ValueError("Labels file is empty")
    
    # Validate images
    missing_images = []
    for img_name in labels.keys():
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            missing_images.append(img_name)
        elif not os.access(img_path, os.R_OK):
            raise PermissionError(f"Cannot read image file: {img_path}")
            
    if missing_images:
        raise ValueError(f"Missing {len(missing_images)} images. First few missing: {missing_images[:5]}")
    
    # Validate class names
    class_names = set(labels.values())
    if not class_names:
        raise ValueError("No classes found in labels file")
    
    print(f"Data directory validated successfully:")
    print(f"- Found {len(labels)} images")
    print(f"- Found {len(class_names)} classes: {sorted(class_names)}")
    return True

class ImageLabelDataset(Dataset):
    def __init__(self, images_dir, class_to_idx, samples=None, labels_json=None, transform=None):
        if samples is not None:
            self.samples = samples  # list of (filename, class_name)
        elif labels_json is not None:
            with open(labels_json, 'r') as f:
                labels = json.load(f)
            self.samples = list(labels.items())
        else:
            raise ValueError("Either samples or labels_json must be provided.")
            
        self.images_dir = images_dir
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            img_name, class_name = self.samples[idx]
            img_path = os.path.join(self.images_dir, img_name)
            
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
                
            image = Image.open(img_path).convert('RGB')
            label = self.class_to_idx[class_name]
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path} at index {idx}. Error: {str(e)}")

def get_class_names_from_labels(labels_json):
    with open(labels_json, 'r') as f:
        labels = json.load(f)
    class_names = sorted(list(set(labels.values())))
    return class_names

def train(data_dir, output_dir, epochs=10, batch_size=32, learning_rate=0.001):
    print("Validating data directory structure...")
    validate_data_directory(data_dir)
    
    print("Setting up training environment...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    images_dir = os.path.join(data_dir, 'images')
    labels_json = os.path.join(data_dir, 'image_labels.json')
    
    print("Loading class names...")
    class_names = get_class_names_from_labels(labels_json)
    num_classes = len(class_names)
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}
    print(f"Found {num_classes} classes: {class_names}")

    # Save class names without explicit permissions
    class_names_file = os.path.join(output_dir, 'class_names.json')
    with open(class_names_file, 'w') as f:
        json.dump(class_names, f)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load all samples and split into train/val
    with open(labels_json, 'r') as f:
        all_items = list(json.load(f).items())
    np.random.shuffle(all_items)
    val_split = 0.2
    split_idx = int(len(all_items) * (1 - val_split))
    train_items = all_items[:split_idx]
    val_items = all_items[split_idx:]

    train_dataset = ImageLabelDataset(images_dir, class_to_idx=class_to_idx, samples=train_items, transform=train_transform)
    val_dataset = ImageLabelDataset(images_dir, class_to_idx=class_to_idx, samples=val_items, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = get_model(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    writer = SummaryWriter(os.path.join(output_dir, 'runs'))
    best_val_acc = 0.0
    best_model_path = os.path.join(output_dir, 'best_model.pth')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if i % 50 == 49:
                train_acc = 100 * correct / total
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], '
                      f'Loss: {running_loss/50:.4f}, Accuracy: {train_acc:.2f}%')
                writer.add_scalar('training loss', running_loss/50, epoch * len(train_loader) + i)
                writer.add_scalar('training accuracy', train_acc, epoch * len(train_loader) + i)
                running_loss = 0.0
                correct = 0
                total = 0
        model.eval()
        val_correct = 0
        val_total = 0
        val_predictions = []
        val_targets = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_predictions.extend(predicted.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        val_acc = 100 * val_correct / val_total
        print(f'Epoch {epoch} - Validation Accuracy: {val_acc:.2f}%')
        report = classification_report(val_targets, val_predictions,
                                    target_names=class_names,
                                    output_dict=True,
                                    zero_division=0)
        metrics_df = pd.DataFrame(report).transpose()
        metrics_df.to_csv(os.path.join(output_dir, f'metrics_epoch_{epoch+1}.csv'))
        scheduler.step(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': best_val_acc,
                'class_names': class_names,
                'num_classes': num_classes
            }, best_model_path)
    metrics = {
        'final_val_accuracy': float(best_val_acc),
        'epochs_trained': epochs,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'num_classes': num_classes,
        'class_names': class_names
    }
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)
    writer.close()
    return best_model_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = train(args.data_dir, args.output_dir, args.epochs, args.batch_size, args.learning_rate)
    print(model_path)