
# 1. IMPORT THE REQUIRED LIBRARIES

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt


# 2. DEFINE PROJECT PATHS

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "digit_recognition.pth"


# 3. REBUILD THE NEURAL NETWORK ARCHITECTURE

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


# 4. CHECK IF THE TRAINED MODEL EXISTS

print("Model location:", MODEL_PATH)
print("Model exists:", MODEL_PATH.exists())

if not MODEL_PATH.exists():
    raise FileNotFoundError("Trained model not found!")


# 5. LOAD THE TRAINED MODEL

model = DigitRecognitionModel()

model.load_state_dict(
    torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
)


# 6. SET THE MODEL TO EVALUATION MODE

model.eval()

print("Trained model loaded successfully!")


# 7. CREATE AN IMPROVED FUNCTION TO PREDICT HANDWRITTEN DIGITS

def predict_digit(image_path):

    from PIL import ImageOps, ImageStat

    # 1. Open image and convert to grayscale
    image = Image.open(image_path).convert("L")

    # 2. Invert image if background is white
    if ImageStat.Stat(image).mean[0] > 127:
        image = ImageOps.invert(image)

    # 3. Crop around the handwritten digit
    bounding_box = image.getbbox()

    if bounding_box is None:
        raise ValueError("No handwritten digit found in image!")

    image = image.crop(bounding_box)

    # 4. Resize while preserving original proportions
    image.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # 5. Create a black 28 x 28 canvas
    canvas = Image.new("L", (28, 28), 0)

    # 6. Calculate position to centre the digit
    x = (28 - image.width) // 2
    y = (28 - image.height) // 2

    # 7. Place digit on canvas
    canvas.paste(image, (x, y))

    # 8. EXPERIMENT: THICKEN THE DIGIT STROKES
    canvas = canvas.filter(ImageFilter.MaxFilter(3))

    # 9. Calculate the centre of mass
    pixels = np.array(canvas, dtype=np.float32)

    total_mass = pixels.sum()

    if total_mass == 0:
        raise ValueError("Image contains no visible digit!")

    y_coordinates, x_coordinates = np.indices(pixels.shape)

    center_x = (pixels * x_coordinates).sum() / total_mass
    center_y = (pixels * y_coordinates).sum() / total_mass

    # 10. Calculate movement towards the centre
    shift_x = 13.5 - center_x
    shift_y = 13.5 - center_y

    # 11. Align the digit using centre of mass
    canvas = canvas.transform(
        (28, 28),
        Image.Transform.AFFINE,
        (1, 0, -shift_x, 0, 1, -shift_y),
        resample=Image.Resampling.BICUBIC,
        fillcolor=0
    )

    # 12. Convert image into a PyTorch tensor
    image_tensor = torch.frombuffer(
        bytearray(canvas.tobytes()),
        dtype=torch.uint8
    ).float().reshape(1, 1, 28, 28) / 255.0

    # 13. Make prediction
    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        top_probabilities, top_digits = torch.topk(
            probabilities, 3, dim=1
        )

        predicted_digit = top_digits[0, 0].item()

    # 14. Display prediction and top 3 results
    print("Predicted digit:", predicted_digit)

    print("\nTop 3 predictions:")

    for i in range(3):

        digit = top_digits[0, i].item()

        confidence = top_probabilities[0, i].item() * 100

        print(f"Digit {digit}: {confidence:.2f}%")

    # 15. Display processed image
    plt.imshow(canvas, cmap="gray")

    plt.title(f"Predicted Digit: {predicted_digit}")

    plt.axis("off")

    plt.show()

    return predicted_digit


# 8. AUTOMATIC MODEL EVALUATION

DIGIT_DIR = BASE_DIR / "data" / "MNIST" / "my_digits"

results = []

for actual_digit in range(10):

    IMAGE_PATH = DIGIT_DIR / f"test_{actual_digit}.png"

    print("\n----------------------------")
    print("Actual digit:", actual_digit)

    predicted_digit = predict_digit(IMAGE_PATH)

    results.append((actual_digit, predicted_digit))


# 9. CALCULATE MODEL ACCURACY

correct = 0

for actual, predicted in results:

    if actual == predicted:
        correct += 1

accuracy = (correct / len(results)) * 100


# 10. DISPLAY FINAL REPORT

print("\n============================")
print("MODEL EVALUATION REPORT")
print("============================")

print("Total images:", len(results))
print("Correct predictions:", correct)
print("Incorrect predictions:", len(results) - correct)

print(f"Final accuracy: {accuracy:.2f}%")
