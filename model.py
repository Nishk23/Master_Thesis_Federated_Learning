
import torch
import torch.nn as nn
import torchvision.models as models

# Helper to auto-detect flattened size
class FlattenedSizeDetector(nn.Module):
    def __init__(self, feature_extractor):
        super().__init__()
        self.features = feature_extractor

    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)
            return x.view(x.size(0), -1).size(1)

# ------------------------------
# 1. TinyCNN (Input: 128x128)
# ------------------------------
class TinyCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(TinyCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [8, 64, 64]
        )
        # Auto-detect flattened size
        dummy_input = torch.zeros(1, 3, 128, 128)
        flat_size = self.features(dummy_input).view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# ------------------------------
# 2. SimpleCNN_V2 (Input: 128x128)
# ------------------------------
class SimpleCNN_V2(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleCNN_V2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),  # -> [32, 64, 64]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),  # -> [64, 32, 32]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # -> [128, 1, 1]
        )

        dummy_input = torch.zeros(1, 3, 128, 128)
        flat_size = self.features(dummy_input).view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(flat_size, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# ------------------------------
# 3. ResNet-18
# ------------------------------
class ResNet18(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# ------------------------------
# 4. MobileNetV2
# ------------------------------
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=3):
        super(MobileNetV2, self).__init__()
        self.model = models.mobilenet_v2(pretrained=False)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

    def forward(self, x):
        return self.model(x)
