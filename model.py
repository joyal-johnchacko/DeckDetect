import torch
import torch.nn as nn
from torchvision import models


class CardClassifier(nn.Module):
    def __init__(self, num_classes=53):
        super(CardClassifier, self).__init__()

        self.base_model = models.mobilenet_v2(weights="IMAGENET1K_V1")

        # ✅ Unfreeze LAST 5 layers so model learns card features better
        layers = list(self.base_model.features.children())
        for layer in layers[:-5]:
            for param in layer.parameters():
                param.requires_grad = False
        for layer in layers[-5:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Replace classifier
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)