import torch
import torch.nn as nn


class StudentModel(nn.Module):
    """
    Student model for non-expert hospitals.
    Lightweight CNN — fewer parameters, faster training,
    designed to learn from both local data and teacher soft predictions.

    Architecture:
        3 Conv blocks → Global Average Pooling → FC layer
    """

    def __init__(self, num_classes=2):
        super(StudentModel, self).__init__()

        # Block 1: 3 → 32 channels
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)          # 224 → 112
        )

        # Block 2: 32 → 64 channels
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)          # 112 → 56
        )

        # Block 3: 64 → 128 channels
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # Block 4: 128 → 256 channels (new)
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.dropout = nn.Dropout(0.4)
        self.fc      = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    import torch

    model = StudentModel(num_classes=2)
    print("Student Model (Lightweight CNN)")
    print("=" * 40)
    print(f"Trainable parameters: {model.get_num_parameters():,}")

    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)

    print(f"Input shape:  {tuple(dummy_input.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Output (logits): {output.detach().numpy().round(3)}")
    print("=" * 40)
    print("Student model working correctly.")