
# ==========================================
# PROJECT 01: HANDWRITTEN DIGIT RECOGNITION
# Training our first model: Neural Network
# ==========================================


# 1. IMPORT REQUIRED LIBRARIES

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt


# 2. CONFIGURATION

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5

DEVICE = torch.device("cpu")


# 3. LOAD THE MNIST DATASET

train_transform = transforms.Compose([

    transforms.RandomRotation(degrees=10),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1)
    ),

    transforms.ToTensor()
])

test_transform = transforms.ToTensor()


train_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=True,
    download=False,
    transform=train_transform
)


test_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=False,
    download=False,
    transform=test_transform
)


# 4. CREATE DATA LOADERS

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# 5. BUILD THE FEEDFORWARD NEURAL NETWORK

class DigitRecognitionModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.flatten = nn.Flatten()

        self.network = nn.Sequential(

            nn.Linear(784, 128),

            nn.ReLU(),

            nn.Linear(128, 10)
        )

    def forward(self, x):

        x = self.flatten(x)

        return self.network(x)


# 6. INITIALISE THE MODEL

model = DigitRecognitionModel().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# 7. TRAIN THE MODEL

print("Starting neural network training...")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}], "
        f"Loss: {average_loss:.4f}"
    )


# 8. EVALUATE THE MODEL

model.eval()

correct = 0
total = 0

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        predictions = torch.argmax(outputs, dim=1)

        total += labels.size(0)

        correct += (predictions == labels).sum().item()

        all_labels.extend(labels.cpu().numpy())

        all_predictions.extend(predictions.cpu().numpy())


accuracy = 100 * correct / total

print("\nMODEL EVALUATION")
print("-------------------------")

print(f"Correct predictions: {correct}/{total}")

print(f"Test accuracy: {accuracy:.2f}%")


# 9. GENERATE CONFUSION MATRIX

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nCONFUSION MATRIX")

print(cm)


# 10. GENERATE CLASSIFICATION REPORT

report = classification_report(
    all_labels,
    all_predictions,
    labels=range(10),
    target_names=[str(i) for i in range(10)],
    digits=4
)

print("\nCLASSIFICATION REPORT")

print("-------------------------")

print(report)


# 11. DISPLAY CONFUSION MATRIX

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=range(10)
)

display.plot(cmap="Blues")

plt.title("Original Neural Network Confusion Matrix")

plt.tight_layout()

plt.show()


# 12. SAVE THE TRAINED MODEL

MODEL_DIR.mkdir(parents=True, exist_ok=True)

model_path = MODEL_DIR / "digit_recognition.pth"

torch.save(model.state_dict(), model_path)

print(f"\nModel saved successfully: {model_path}")
