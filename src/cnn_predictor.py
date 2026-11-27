
# ==========================================
# PROJECT 01: HANDWRITTEN DIGIT RECOGNITION
# Reusable CNN Prediction Engine
# Built and Designed by @uchecyril
# ==========================================


# 1. IMPORT REQUIRED LIBRARIES

from pathlib import Path

import numpy as np

import torch
import torch.nn as nn

from PIL import (
    Image,
    ImageFilter,
    ImageOps,
    ImageStat,
    ImageEnhance
)


# 2. DEFINE PROJECT PATHS

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "cnn_digit_recognition.pth"


# 3. REBUILD THE TRAINED CNN ARCHITECTURE

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


# 4. LOAD THE TRAINED CNN MODEL

def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Trained CNN model not found: {MODEL_PATH}"
        )

    model = DigitRecognitionCNN()

    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state_dict)

    model.eval()

    return model


# 5. PREPROCESS THE USER'S IMAGE


def preprocess_image(image):

    # 1. Convert the image to grayscale
    image = image.convert("L")

    # 2. Invert the image if the background is white
    if ImageStat.Stat(image).mean[0] > 127:
        image = ImageOps.invert(image)

    # 3. Find and crop the handwritten digit
    bounding_box = image.getbbox()

    if bounding_box is None:
        raise ValueError(
            "No handwritten digit found. Please draw a digit."
        )

    image = image.crop(bounding_box)

    # 4. Resize the digit while preserving its proportions
    image.thumbnail(
        (20, 20),
        Image.Resampling.LANCZOS
    )

    # 5. Place the digit on a black 28 x 28 canvas
    canvas = Image.new("L", (28, 28), 0)

    x = (28 - image.width) // 2
    y = (28 - image.height) // 2

    canvas.paste(image, (x, y))

    # 6. Calculate the centre of mass
    pixels = np.array(canvas, dtype=np.float32)

    total_mass = pixels.sum()

    if total_mass == 0:
        raise ValueError(
            "Image contains no visible digit."
        )

    y_coordinates, x_coordinates = np.indices(
        pixels.shape
    )

    center_x = (
        (pixels * x_coordinates).sum() / total_mass
    )

    center_y = (
        (pixels * y_coordinates).sum() / total_mass
    )

    # 7. Align the digit towards the centre
    shift_x = 13.5 - center_x
    shift_y = 13.5 - center_y

    canvas = canvas.transform(
        (28, 28),
        Image.Transform.AFFINE,
        (1, 0, -shift_x, 0, 1, -shift_y),
        resample=Image.Resampling.BICUBIC,
        fillcolor=0
    )

    # 8. Return the processed image
    return canvas



# 6. MAKE A PREDICTION

def predict_image(image, model):

    # Preprocess the user's image
    processed_image = preprocess_image(image)

    # Convert the processed image into a NumPy array
    image_array = np.array(
        processed_image,
        dtype=np.uint8,
        copy=True
    )

    # Convert the array into a PyTorch tensor
    image_tensor = torch.from_numpy(
        image_array
    ).float().unsqueeze(0).unsqueeze(0) / 255.0

    # Make the prediction
    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        top_probabilities, top_digits = torch.topk(
            probabilities,
            3,
            dim=1
        )

    # Get the predicted digit
    predicted_digit = top_digits[0, 0].item()

    # Prepare the top three predictions
    top_predictions = []

    for i in range(3):

        digit = top_digits[0, i].item()

        confidence = (
            top_probabilities[0, i].item() * 100
        )

        top_predictions.append({

            "digit": digit,

            "confidence": confidence
        })

    # Return the prediction results
    return {

        "predicted_digit": predicted_digit,

        "top_predictions": top_predictions,

        "processed_image": processed_image
    }
