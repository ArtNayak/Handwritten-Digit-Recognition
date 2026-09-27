import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class MLP(nn.Module):
    def __init__(self, num_classes=10, p_drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self, num_classes=10, p_drop=0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 28->14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14->7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p_drop),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def transfer_resnet18(num_classes=10):
    """Pretrained resnet18, first conv adapted to 1-channel input.
    Frozen: conv1's original stem is replaced, layer1-3 stay frozen (ImageNet low/mid-level
    features reused as-is). Unfrozen: conv1 (new), layer4 (last block, task-specific
    high-level features), fc (new head) — a step up from head-only fine-tuning."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    old_conv = model.conv1
    model.conv1 = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride, padding=old_conv.padding, bias=False)
    for p in model.parameters():
        p.requires_grad = False
    model.conv1.requires_grad_(True)
    model.layer4.requires_grad_(True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # new head, trainable by default
    return model


def build_model(name, num_classes=10):
    return {"mlp": MLP, "cnn": CNN, "transfer": transfer_resnet18}[name](num_classes=num_classes)
