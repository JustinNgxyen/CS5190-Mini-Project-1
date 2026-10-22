
import torch
import torch.nn as nn
import torch.optim as optim

# Loss function
criterion = nn.CrossEntropyLoss()

model = nn.Sequential(
    # First layer
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    # Second layer
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    # Third layer
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    # Flatten
    nn.Flatten(),

    # Fully connected layer
    nn.Linear(128 * 16 * 16, 128),
    nn.ReLU(),

    # Dropout
    nn.Dropout(0.5),

    # Output layer (Real or Fake)
    nn.Linear(128, 2)
)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
