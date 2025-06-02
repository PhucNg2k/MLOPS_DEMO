import torch
import torch.nn as nn
import torchvision.models as models

class AnimalClassifier(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(AnimalClassifier, self).__init__()
        # Load pretrained ResNet50
        self.resnet = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        
        # Replace the final layer for multi-class classification
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)  # Multiple animal classes
        )

    def forward(self, x):
        return self.resnet(x)

def get_model(num_classes=10):
    """
    Get the animal classification model
    
    Args:
        num_classes (int): Number of animal classes to classify
        
    Returns:
        AnimalClassifier: The model instance
    """
    model = AnimalClassifier(num_classes=num_classes, pretrained=True)
    return model 