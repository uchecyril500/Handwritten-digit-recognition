
# ==========================================
# PROJECT 01: HANDWRITTEN DIGIT RECOGNITION
# Training our second model: CNN
# ==========================================


# 1. IMPORT REQUIRED LIBRARIES

from pathlib import Path

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report


# 2. CONFIGURATION

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5

DEVICE = torch.device("cpu")


# 3. LOAD MNIST DATASET WITH DATA AUGMENTATION

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


# 5. BUILD THE CONVOLUTIONAL NEURAL NETWORK

class DigitRecognitionCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            # First convolutional layer
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            # Reduce image dimensions
            nn.MaxPool2d(kernel_size=2),

            # Second convolutional layer
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            # Reduce image dimensions again
            nn.MaxPool2d(kernel_size=2)
        )


        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(32 * 7 * 7, 128),

            nn.ReLU(),

            nn.Linear(128, 10)
        )


    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# 6. INITIALISE THE CNN

model = DigitRecognitionCNN().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# 7. TRAIN THE CNN

print("Starting CNN training...")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Clear previous gradients
        optimizer.zero_grad()

        # Make predictions
        outputs = model(images)

        # Calculate prediction error
        loss = criterion(outputs, labels)

        # Calculate gradients
        loss.backward()

        # Update model weights
        optimizer.step()

        total_loss += loss.item()


    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}], "
        f"Loss: {average_loss:.4f}"
    )


# 8. EVALUATE THE CNN

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


print("\nCNN MODEL EVALUATION")
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

plt.title("CNN MNIST Confusion Matrix")

plt.tight_layout()

plt.show()


# 12. SAVE THE CNN MODEL

MODEL_DIR.mkdir(parents=True, exist_ok=True)

model_path = MODEL_DIR / "cnn_digit_recognition.pth"

torch.save(model.state_dict(), model_path)

print(f"\nCNN model saved successfully: {model_path}")
