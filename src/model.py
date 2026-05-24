import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=38, pretrained=True):
    weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_v2_s(weights=weights)

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Replace final classifier
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(1280, num_classes)
    )

    return model

if __name__ == "__main__":
    model = get_model(num_classes=38)
    print("✅ Model loaded successfully")
    print(f"✅ Output classes : 38")

    # Test forward pass
    dummy = torch.randn(1, 3, 224, 224)
    out = model(dummy)
    print(f"✅ Output shape   : {out.shape}")