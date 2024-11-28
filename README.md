# MNIST Classification with PyTorch

[![CI/CD Pipeline](https://github.com/anudeep-j98/cnn_training_learinig/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/anudeep-j98/cnn_training_learinig/actions/workflows/ci-cd.yml)

This project implements a convolutional neural network (CNN) for classifying handwritten digits from the MNIST dataset using PyTorch. The model is designed to achieve high accuracy while maintaining a low number of parameters.

## Project Structure

│
├── .github/
│   └── workflows/
│       └── ci-cd.yml
│
├── model/
│   ├── net.py          # Model architecture
│   └── train.py        # Training script
├── tests/
│   └── test_model.py   # Testing script
│
├── requirements.txt     # Required Python packages
└── README.md            # Project documentation


## Model Architecture

The model is defined in `model/net.py` and consists of the following layers:

- **Convolutional Layers**: Four convolutional layers with ReLU activation.
- **Fully Connected Layers**: Two fully connected layers leading to the output layer.

### Model Characteristics

- **Parameters**: Less than 25,000 parameters.
- **Accuracy**: Achieves over 95% accuracy on the MNIST test set in one epoch.

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   ```

2. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the training script**:
   ```bash
   python model/train.py
   ```

4. **Run the tests**:
   ```bash
   python -m unittest discover tests/
   ```

## Testing

The testing script (`tests/test_model.py`) includes the following tests:

1. **Parameter Count**: Checks that the model has less than 25,000 parameters.
2. **Accuracy**: Validates that the model achieves over 95% accuracy on the MNIST test set.
3. **Input Shape Validation**: Ensures the model accepts input of shape `(1, 1, 28, 28)`.
4. **Output Shape Validation**: Verifies that the model outputs a tensor of shape `(1, 10)`.
5. **Inference Time**: Measures the inference time for a single input batch, ensuring it is less than 0.1 seconds.

## CI/CD Pipeline

The project includes a GitHub Actions workflow defined in `.github/workflows/ci-cd.yml`. This workflow will:

- Install dependencies.
- Train the model.
- Run the tests.

The CI/CD pipeline is triggered on every push to the `main` branch.