import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


from CNN import model

data_path = Path("miniproject1/DeepFakeDetection")

real_images = list((data_path / "real").glob("*"))
fake_images = list((data_path / "fake").glob("*"))

image_paths = real_images + fake_images
labels = [0] * len(real_images) + [1] * len(fake_images)

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.30,
    stratify=labels,
    random_state=42
)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=0.50,
    stratify=temp_labels,
    random_state=42
)

class DeepfakeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

evaluation_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

train_dataset = DeepfakeDataset(
    train_paths,
    train_labels,
    train_transform
)

val_dataset = DeepfakeDataset(
    val_paths,
    val_labels,
    evaluation_transform
)

test_dataset = DeepfakeDataset(
    test_paths,
    test_labels,
    evaluation_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

print("Training images:", len(train_loader.dataset))
print("Training batches:", len(train_loader))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10

for epoch in range(epochs):
    model.train()
    training_loss = 0.0
    training_correct = 0
    training_total = 0

    for batch_index, (inputs, targets) in enumerate(train_loader):
    
        optimizer.zero_grad()  # Clear previous gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs,  targets)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        if (batch_index + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_index + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        training_loss += loss.item() * inputs.size(0)
        training_correct += (
            outputs.argmax(dim=1) == targets.view(-1)
        ).sum().item()
        training_total += targets.numel()

    print(f"Epoch {epoch + 1}/{epochs}")
    print(f"Average training loss: {training_loss / training_total:.4f}")
    print(f"Training accuracy: {training_correct / training_total:.4f}")

    model.eval()
    with torch.no_grad():
        validation_loss = 0
        validation_correct = 0
        validation_total = 0
        for batch_index, (inputs, targets) in enumerate(val_loader):
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, targets)  # Compute loss

            validation_loss += loss.item() * inputs.size(0)
            validation_correct += (
                outputs.argmax(dim=1) == targets.view(-1)
            ).sum().item()
            validation_total += targets.numel()

    print(f"Validation loss: {validation_loss / validation_total:.4f}")
    print(f"Validation accuracy: {validation_correct / validation_total:.4f}")

model.eval()
test_loss = 0
test_total = 0
all_targets = []
all_predictions = []

with torch.no_grad():

    for batch_index, (inputs, targets) in enumerate(test_loader):
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        targets = targets.to(device).long().view(-1)
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, targets)  # Compute loss
        predictions = outputs.argmax(dim=1)

        test_loss += loss.item() * inputs.size(0)
        test_total += targets.size(0)
        all_targets.extend(targets.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, pos_label=1, zero_division=0)
    recall = recall_score(all_targets, all_predictions, pos_label=1, zero_division=0)
    f1 = f1_score(all_targets, all_predictions, pos_label=1, zero_division=0)

    cm = confusion_matrix(all_targets, all_predictions, labels=[0, 1])

    # Average test loss
    avg_test_loss = test_loss / test_total

    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test precision: {precision:.4f}")
    print(f"Test recall: {recall:.4f}")
    print(f"Test F1 score: {f1:.4f}")
    print(f"Test loss: {avg_test_loss:.4f}")

    print("Confusion matrix:")
    print(cm)




