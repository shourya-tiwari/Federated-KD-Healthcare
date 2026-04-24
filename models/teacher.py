import torch
import torch.nn as nn
import torchvision.models as models


class TeacherModel(nn.Module):
    """
    Teacher model for expert hospitals.
    Uses pretrained ResNet-18 as backbone — powerful and well-tested
    on medical imaging tasks.
    
    Pretrained on ImageNet, fine-tuned for our binary classification:
    0 = Normal, 1 = Pneumonia
    """

    def __init__(self, num_classes=2, pretrained=True):
        super(TeacherModel, self).__init__()

        # Load pretrained ResNet-18
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # Replace final classification layer
        # Original: Linear(512, 1000) for ImageNet
        # Ours:     Linear(512, 2)    for Normal vs Pneumonia
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    import torch

    model = TeacherModel(num_classes=2, pretrained=True)
    print("Teacher Model (ResNet-18)")
    print("=" * 40)
    print(f"Trainable parameters: {model.get_num_parameters():,}")

    # Test forward pass with dummy input
    # Simulates a batch of 4 images, 3 channels, 224x224
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)

    print(f"Input shape:  {tuple(dummy_input.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Output (logits): {output.detach().numpy().round(3)}")
    print("=" * 40)
    print("Teacher model working correctly.")