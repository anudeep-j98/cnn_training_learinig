# MNIST Classification with PyTorch

[![CI/CD Pipeline](https://github.com/anudeep-j98/cnn_training_learinig/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/anudeep-j98/cnn_training_learinig/actions/workflows/ci-cd.yml)

This project implements a convolutional neural network (CNN) for classifying handwritten digits from the MNIST dataset using PyTorch. The model is designed to achieve high accuracy while maintaining a low number of parameters.

## Model Architecture

The model is defined in `model/net.py` and consists of the following layers:

- **Convolutional Layers**: Four convolutional layers with ReLU activation.
- **BatchNorm** - Twoi layers of batch normalization
- **Fully Connected Layers**: Two fully connected layers leading to the output layer.

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 4, 26, 26]              40
            Conv2d-2            [-1, 8, 24, 24]             296
       BatchNorm2d-3            [-1, 8, 12, 12]              16
            Conv2d-4           [-1, 16, 10, 10]           1,168
            Conv2d-5             [-1, 32, 8, 8]           4,640
       BatchNorm2d-6             [-1, 32, 4, 4]              64
           Flatten-7                  [-1, 512]               0
            Linear-8                   [-1, 35]          17,955
            Linear-9                   [-1, 10]             360
================================================================
Total params: 24,539
Trainable params: 24,539
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.10
Params size (MB): 0.09
Estimated Total Size (MB): 0.20
----------------------------------------------------------------
```


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
