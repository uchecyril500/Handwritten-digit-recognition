
# ==========================================
# PROJECT 01: HANDWRITTEN DIGIT RECOGNITION
# Testing our CNN model on personal handwriting
# ==========================================


# 1. IMPORT REQUIRED LIBRARIES

import numpy as np
import torch
import torch.nn as nn

from pathlib import Path

from PIL import (
    Image,
    ImageFilter,
    ImageOps,
    ImageStat,
    ImageEnhance
)

import matplotlib.pyplot as plt


# 2. DEFINE PROJECT PATHS

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "cnn_digit_recognition.pth"

DIGIT_DIR = BASE_DIR / "data" / "MNIST" / "my_digits"


# 3. REBUILD THE CNN ARCHITECTURE

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


# 4. CHECK IF THE TRAINED CNN EXISTS

print("CNN model location:", MODEL_PATH)

print("CNN model exists:", MODEL_PATH.exists())

if not MODEL_PATH.exists():

    raise FileNotFoundError("CNN model not found!")


# 5. LOAD THE TRAINED CNN

model = DigitRecognitionCNN()

model.load_state_dict(

    torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )
)

model.eval()

print("Trained CNN loaded successfully!")


# 6. CREATE THE DIGIT PREDICTION FUNCTION

def predict_digit(image_path):

    # Open image and convert to grayscale
    image = Image.open(image_path).convert("L")

    # Invert image if background is white
    if ImageStat.Stat(image).mean[0] > 127:

        image = ImageOps.invert(image)

    # Crop around the handwritten digit
    bounding_box = image.getbbox()

    if bounding_box is None:

        raise ValueError("No handwritten digit found!")

    image = image.crop(bounding_box)

    # Resize while preserving proportions
    image.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # Create black 28 x 28 canvas
    canvas = Image.new("L", (28, 28), 0)

    # Centre the digit
    x = (28 - image.width) // 2

    y = (28 - image.height) // 2

    canvas.paste(image, (x, y))

    # Thicken the digit strokes
    canvas = canvas.filter(ImageFilter.MaxFilter(3))

    # Calculate centre of mass
    pixels = np.array(canvas, dtype=np.float32)

    total_mass = pixels.sum()

    if total_mass == 0:

        raise ValueError("Image contains no visible digit!")

    y_coordinates, x_coordinates = np.indices(pixels.shape)

    center_x = (pixels * x_coordinates).sum() / total_mass

    center_y = (pixels * y_coordinates).sum() / total_mass

    # Calculate movement towards the centre
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

    # Increase image brightness
    enhancer = ImageEnhance.Brightness(canvas)

    canvas = enhancer.enhance(3.0)

    # Keep pixel values within valid grayscale range
    canvas = canvas.point(lambda p: min(255, p))

    # Convert image into a PyTorch tensor
    image_tensor = torch.frombuffer(

        bytearray(canvas.tobytes()),

        dtype=torch.uint8

    ).float().reshape(1, 1, 28, 28) / 255.0

    # Make prediction
    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        top_probabilities, top_digits = torch.topk(

            probabilities,

            3,

            dim=1
        )

        predicted_digit = top_digits[0, 0].item()


    # Display results
    print("Predicted digit:", predicted_digit)

    print("\nTop 3 predictions:")

    for i in range(3):

        digit = top_digits[0, i].item()

        confidence = top_probabilities[0, i].item() * 100

        print(f"Digit {digit}: {confidence:.2f}%")


    # Display processed image
    plt.figure(figsize=(4, 4))

    plt.imshow(canvas, cmap="gray", vmin=0, vmax=255)

    plt.title(f"CNN Prediction: {predicted_digit}")

    plt.axis("off")

    plt.show()

    return predicted_digit


# 7. CHECK THAT PERSONAL HANDWRITING IMAGES EXIST

if not DIGIT_DIR.exists():

    raise FileNotFoundError(
        f"Handwritten digit directory not found: {DIGIT_DIR}"
    )


# 8. TEST ALL TEN HANDWRITTEN DIGITS

results = []

for actual_digit in range(10):

    IMAGE_PATH = DIGIT_DIR / f"test_{actual_digit}.png"

    if not IMAGE_PATH.exists():

        raise FileNotFoundError(
            f"Handwritten image not found: {IMAGE_PATH}"
        )

    print("\n----------------------------")

    print("Actual digit:", actual_digit)

    predicted_digit = predict_digit(IMAGE_PATH)

    results.append((actual_digit, predicted_digit))


# 9. CALCULATE CNN ACCURACY ON CUSTOM IMAGES

correct = sum(

    actual == predicted

    for actual, predicted in results
)

accuracy = (correct / len(results)) * 100


# 10. DISPLAY FINAL EVALUATION REPORT

print("\n============================")

print("CNN MODEL EVALUATION REPORT")

print("============================")

print("Total images:", len(results))

print("Correct predictions:", correct)

print("Incorrect predictions:", len(results) - correct)

print(f"Final accuracy: {accuracy:.2f}%")
