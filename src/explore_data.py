
# 1. IMPORT REQUIRED LIBRARIES

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from torchvision import datasets, transforms
from PIL import Image, ImageOps, ImageStat, ImageFilter


# 2. LOCATE THE DATASET AND CUSTOM IMAGE

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

CUSTOM_IMAGE = DATA_DIR / "MNIST" / "my_digits" / "test_3.png"


# 3. LOAD MNIST TRAINING DATASET

train_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=True,
    download=False,
    transform=transforms.ToTensor()
)


# 4. COLLECT FIVE EXAMPLES OF DIGITS 3 AND 5

digit_3_images = []
digit_5_images = []

for image, label in train_dataset:

    if label == 3 and len(digit_3_images) < 5:
        digit_3_images.append(image)

    elif label == 5 and len(digit_5_images) < 5:
        digit_5_images.append(image)

    if len(digit_3_images) == 5 and len(digit_5_images) == 5:
        break


# 5. PREPROCESS YOUR HANDWRITTEN DIGIT
# Use the same preprocessing as predict.py

def preprocess_custom_digit(image_path):

    image = Image.open(image_path).convert("L")

    # Invert white background
    if ImageStat.Stat(image).mean[0] > 127:
        image = ImageOps.invert(image)

    # Crop around the digit
    bounding_box = image.getbbox()

    if bounding_box is None:
        raise ValueError("No handwritten digit found!")

    image = image.crop(bounding_box)

    # Resize while preserving proportions
    image.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # Place on black 28 x 28 canvas
    canvas = Image.new("L", (28, 28), 0)

    x = (28 - image.width) // 2
    y = (28 - image.height) // 2

    canvas.paste(image, (x, y))

    # Apply the same stroke thickening
    canvas = canvas.filter(ImageFilter.MaxFilter(3))

    # Calculate centre of mass
    pixels = np.array(canvas, dtype=np.float32)

    total_mass = pixels.sum()

    if total_mass == 0:
        raise ValueError("Image contains no visible digit!")

    y_coordinates, x_coordinates = np.indices(pixels.shape)

    center_x = (pixels * x_coordinates).sum() / total_mass
    center_y = (pixels * y_coordinates).sum() / total_mass

    shift_x = 13.5 - center_x
    shift_y = 13.5 - center_y

    # Align the digit
    canvas = canvas.transform(
        (28, 28),
        Image.Transform.AFFINE,
        (1, 0, -shift_x, 0, 1, -shift_y),
        resample=Image.Resampling.BICUBIC,
        fillcolor=0
    )

    return canvas


# 6. PROCESS YOUR HANDWRITTEN DIGIT 3

custom_image = preprocess_custom_digit(CUSTOM_IMAGE)


# 7. DISPLAY THE COMPARISON

fig, axes = plt.subplots(3, 5, figsize=(12, 7))


# First row: MNIST digit 3

for i in range(5):

    axes[0, i].imshow(
        digit_3_images[i].squeeze(0),
        cmap="gray"
    )

    axes[0, i].set_title("MNIST: 3")

    axes[0, i].axis("off")


# Second row: MNIST digit 5

for i in range(5):

    axes[1, i].imshow(
        digit_5_images[i].squeeze(0),
        cmap="gray"
    )

    axes[1, i].set_title("MNIST: 5")

    axes[1, i].axis("off")


# Third row: Your processed handwritten 3

axes[2, 0].imshow(custom_image, cmap="gray")

axes[2, 0].set_title("Your processed 3")

axes[2, 0].axis("off")


# Hide remaining spaces

for i in range(1, 5):

    axes[2, i].axis("off")


# 8. DISPLAY RESULTS

plt.suptitle(
    "MNIST Digits 3 and 5 vs My Processed Handwriting"
)

plt.tight_layout()

plt.show()
