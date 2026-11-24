
from pathlib import Path
from torchvision import datasets, transforms

# 1. Define where our dataset will be stored
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# 2. Convert images into PyTorch tensors
transform = transforms.ToTensor()

# 3. Download the training dataset
train_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform
)

# 4. Download the testing dataset
test_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform
)

# 5. Display dataset information
print("MNIST DATASET DOWNLOADED SUCCESSFULLY!")

print("Training images:", len(train_dataset))
print("Testing images:", len(test_dataset))

# 6. Inspect the first training image
image, label = train_dataset[0]

print("Image shape:", image.shape)
print("Image label:", label)
