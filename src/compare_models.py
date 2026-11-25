
# ==========================================
# PROJECT 01: HANDWRITTEN DIGIT RECOGNITION
# COMPARING OUR TWO TRAINED MODELS
# ==========================================


# 1. IMPORT REQUIRED LIBRARIES

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from PIL import (
    Image,
    ImageOps,
    ImageStat,
    ImageFilter,
    ImageEnhance
)


# 2. DEFINE PROJECT PATHS

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"

DIGIT_DIR = BASE_DIR / "data" / "MNIST" / "my_digits"

RESULTS_DIR = BASE_DIR / "results"

ORIGINAL_MODEL_PATH = MODEL_DIR / "digit_recognition.pth"

CNN_MODEL_PATH = MODEL_DIR / "cnn_digit_recognition.pth"


# 3. CHECK THAT BOTH MODELS EXIST

if not ORIGINAL_MODEL_PATH.exists():
    raise FileNotFoundError("Original model not found!")

if not CNN_MODEL_PATH.exists():
    raise FileNotFoundError("CNN model not found!")


# 4. DEFINE THE ORIGINAL NEURAL NETWORK

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


# 5. DEFINE THE CNN

class DigitRecognitionCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 16, kernel_size=3, padding=1),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),

            nn.ReLU(),

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


# 6. LOAD BOTH TRAINED MODELS

original_model = DigitRecognitionModel()

original_model.load_state_dict(

    torch.load(
        ORIGINAL_MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )
)

original_model.eval()


cnn_model = DigitRecognitionCNN()

cnn_model.load_state_dict(

    torch.load(
        CNN_MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )
)

cnn_model.eval()


print("Both trained models loaded successfully!")


# 7. PREPROCESS HANDWRITTEN IMAGES

def preprocess_image(image_path):

    # Open image and convert to grayscale
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

    # Create black 28 x 28 canvas
    canvas = Image.new("L", (28, 28), 0)

    x = (28 - image.width) // 2

    y = (28 - image.height) // 2

    canvas.paste(image, (x, y))

    # Thicken digit strokes
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

    # ==========================================
    # INCREASE IMAGE BRIGHTNESS
    # ==========================================

    enhancer = ImageEnhance.Brightness(canvas)

    canvas = enhancer.enhance(3.0)

    # Keep pixel values within valid grayscale range
    canvas = canvas.point(lambda p: min(255, p))

    # ==========================================
    # CONVERT TO PYTORCH TENSOR
    # ==========================================

    image_tensor = torch.frombuffer(

        bytearray(canvas.tobytes()),

        dtype=torch.uint8

    ).float().reshape(1, 1, 28, 28) / 255.0

    return canvas, image_tensor


# 8. COMPARE BOTH MODELS ON ALL TEN DIGITS

results = []

print("\nMODEL COMPARISON RESULTS")
print("============================")


for actual_digit in range(10):

    image_path = DIGIT_DIR / f"test_{actual_digit}.png"

    processed_image, image_tensor = preprocess_image(image_path)

    # Get predictions from both models
    with torch.no_grad():

        original_outputs = original_model(image_tensor)

        cnn_outputs = cnn_model(image_tensor)

        original_probabilities = torch.softmax(
            original_outputs, dim=1
        )

        cnn_probabilities = torch.softmax(
            cnn_outputs, dim=1
        )

        original_prediction = torch.argmax(
            original_probabilities, dim=1
        ).item()

        cnn_prediction = torch.argmax(
            cnn_probabilities, dim=1
        ).item()

    # Store results
    results.append({

        "actual": actual_digit,

        "original": original_prediction,

        "cnn": cnn_prediction,

        "image": processed_image
    })

    print(
        f"Actual: {actual_digit} | "
        f"Original: {original_prediction} | "
        f"CNN: {cnn_prediction}"
    )


# 9. CALCULATE ACCURACY FOR BOTH MODELS

original_correct = sum(

    result["actual"] == result["original"]

    for result in results
)

cnn_correct = sum(

    result["actual"] == result["cnn"]

    for result in results
)


original_accuracy = original_correct / len(results) * 100

cnn_accuracy = cnn_correct / len(results) * 100


# 10. DISPLAY VISUAL COMPARISON

fig, axes = plt.subplots(2, 5, figsize=(15, 9))

fig.suptitle(
    "Original Neural Network vs CNN",
    fontsize=16
)


for i, result in enumerate(results):

    ax = axes.flat[i]

    ax.imshow(result["image"], cmap="gray", vmin=0, vmax=255)

    actual = result["actual"]

    original = result["original"]

    cnn = result["cnn"]

    ax.set_title(

        f"Actual: {actual}\n"
        f"Original: {original}\n"
        f"CNN: {cnn}",

        fontsize=10
    )

    ax.axis("off")


plt.subplots_adjust(
    top=0.88,
    bottom=0.08,
    hspace=0.65,
    wspace=0.25
)


# 11. SAVE THE VISUAL REPORT

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

report_path = RESULTS_DIR / "model_comparison.png"

plt.savefig(
    report_path,
    dpi=200,
    bbox_inches="tight"
)

print("\nVISUAL REPORT SAVED:")
print(report_path)


# 12. DISPLAY FINAL COMPARISON

print("\n============================")

print("FINAL MODEL COMPARISON")

print("============================")

print(
    f"Original network: "
    f"{original_correct}/10 "
    f"({original_accuracy:.2f}%)"
)

print(
    f"CNN: "
    f"{cnn_correct}/10 "
    f"({cnn_accuracy:.2f}%)"
)

plt.show()
