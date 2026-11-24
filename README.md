# Handwritten Digit Recognition Using Neural Networks and CNNs

## 1. Project Overview

This project develops an artificial intelligence system that recognises handwritten digits (0–9) using deep learning.

The project uses the MNIST dataset to train and evaluate two neural network architectures:

1. A traditional feedforward neural network.
2. A Convolutional Neural Network (CNN).

Both models are also tested using ten handwritten digit images created manually.

The main objective is to understand how neural network architectures learn image features and compare their performance on digit recognition.

## 2. Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Pillow (PIL)
- Scikit-learn

## 3. Dataset

The project uses the MNIST handwritten digit dataset.

MNIST contains:
- 60,000 training images.
- 10,000 testing images.
- 10 different digit classes (0–9).
- Grayscale images of size 28 × 28 pixels.

In addition to MNIST, ten personally handwritten digit images were created and stored in `data/MNIST/my_digits/`.

These images were used to test how well the trained models recognise handwriting outside the original dataset.

## 4. Data Preprocessing and Augmentation

The training images were prepared using PyTorch transforms.

Data augmentation was applied to the training dataset using:

- Random rotation of up to 10 degrees.
- Random translation of up to 10%.
- Random scaling between 90% and 110%.

The testing dataset was kept unchanged.

For personal handwritten images, preprocessing included:

- Conversion to grayscale.
- Background inversion when necessary.
- Cropping around the digit.
- Resizing while preserving proportions.
- Placement on a 28 × 28 black background.
- Centre-of-mass alignment.
- Conversion into a PyTorch tensor.

These steps help make personal handwriting images more compatible with the MNIST training format.

## 5. Model Architectures

### Model 1: Feedforward Neural Network

The first model uses a simple feedforward neural network.

Architecture:

- Flatten layer: converts the 28 × 28 image into 784 input values.
- Fully connected layer: 784 to 128 neurons.
- ReLU activation.
- Output layer: 128 to 10 neurons.

The model produces scores for the ten possible digit classes.

### Model 2: Convolutional Neural Network (CNN)

The second model uses convolutional layers to learn image features.

Architecture:

- Convolutional layer: 1 input channel, 16 output channels.
- ReLU activation.
- Max pooling.
- Convolutional layer: 16 input channels, 32 output channels.
- ReLU activation.
- Max pooling.
- Flatten layer.
- Fully connected layer: 1568 to 128 neurons.
- ReLU activation.
- Output layer: 128 to 10 neurons.

The CNN learns visual patterns such as edges, curves and shapes that help distinguish handwritten digits.

## 6. Training Configuration

Both models were trained using the following configuration:

| Parameter | Value |
|---|---|
| Dataset | MNIST |
| Batch size | 64 |
| Learning rate | 0.001 |
| Training epochs | 5 |
| Optimizer | Adam |
| Loss function | CrossEntropyLoss |
| Device | CPU |

## 7. Model Evaluation

The models were evaluated using the MNIST test dataset and ten personally handwritten digit images.

The following metrics were used:

- Accuracy
- Confusion matrix
- Precision
- Recall
- F1-score

### MNIST Test Results

| Model | Test Accuracy |
|---|---:|
| Feedforward Neural Network | 96.97% |
| CNN | 98.82% |

The CNN achieved higher accuracy on the MNIST test dataset.

### Personal Handwriting Results

| Model | Correct Predictions | Accuracy |
|---|---:|---:|
| Feedforward Neural Network | 9/10 | 90% |
| CNN | 10/10 | 100% |

The original neural network incorrectly classified the handwritten digit 3 as 5.

The CNN correctly classified all ten handwritten test images.

**Important:** The personal handwriting evaluation contains only ten images. The 100% result applies to these ten samples and does not establish general performance on unseen handwriting.

## 8. Visual Model Comparison

The following image shows the predictions made by both models on the ten personal handwritten digits.

![Original Neural Network vs CNN](results/model_comparison.png)

The comparison demonstrates the improvement achieved by the CNN in this experiment.

## 9. Project Structure

```text
01-Handwritten-Digit-Recognition/
│
├── data/
│   └── MNIST/
│       ├── my_digits/
│       └── raw/
│
├── models/
│   ├── digit_recognition.pth
│   └── cnn_digit_recognition.pth
│
├── notebooks/
│
├── results/
│   └── model_comparison.png
│
├── src/
│   ├── download_data.py
│   ├── explore_data.py
│   ├── train.py
│   ├── predict.py
│   ├── predict_cnn.py
│   └── compare_models.py
│
└── README.md
```

## 10. How to Run the Project

### Step 1: Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Enter the project directory

```bash
cd 01-Handwritten-Digit-Recognition
```

### Step 3: Create a virtual environment

```bash
python -m venv .venv
```

### Step 4: Activate the virtual environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 5: Install the required libraries

```bash
pip install torch torchvision numpy matplotlib pillow scikit-learn
```

### Step 6: Download the MNIST dataset if necessary

```bash
python src/download_data.py
```

### Step 7: Train the original neural network

```bash
python src/train.py
```

### Step 8: Train the CNN

The CNN training implementation is included in the training script. Follow the CNN training configuration in the source code.

### Step 9: Run digit predictions

```bash
python src/predict.py
```

For CNN predictions:

```bash
python src/predict_cnn.py
```

### Step 10: Compare both models

```bash
python src/compare_models.py
```

## 11. What I Learned

Through this project, I gained practical experience in:

- Loading and exploring image datasets.
- Preparing images for machine learning.
- Applying data augmentation.
- Building neural networks using PyTorch.
- Training and evaluating deep learning models.
- Understanding convolutional neural networks.
- Using confusion matrices and classification reports.
- Comparing different model architectures.
- Testing models using personally created images.
- Improving image preprocessing for prediction.

## 12. Conclusion

This project demonstrates the development and evaluation of two deep learning models for handwritten digit recognition.

The CNN achieved better results than the original feedforward neural network on both the MNIST test dataset and the ten personal handwritten samples used in this experiment.

The project provided practical experience in the machine learning workflow, from data preparation and model development to evaluation and visualisation.

**Project:** AI Engineering Portfolio — Project 01  
**Topic:** Handwritten Digit Recognition  
**Framework:** PyTorch