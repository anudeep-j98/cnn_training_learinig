import torch
import torch.nn as nn
from torchvision import datasets, transforms
from model.net import Net
import time

# Load MNIST dataset for testing
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ])

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=2, shuffle=False)

def test_model():
    model = Net()
    model.load_state_dict(torch.load('model/mnist_model.pth'))
    model.eval()

    # Test 1: Check number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    assert num_params < 25000, f"Model has {num_params} parameters, exceeds limit."

    # Test 2: Input shape validation
    batch_data, _ = next(iter(test_loader))  # Batch size of 2, 1 channel, 28x28 image
    assert batch_data.shape == (2, 1, 28, 28), f"Input shape is {batch_data[0].shape}, expected (2, 1, 28, 28)."

    # Test 3: Check accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = correct / total
    assert accuracy > 0.95, f"Model accuracy is {accuracy}, expected > 0.95."

    # Test 4: Output shape validation
    with torch.no_grad():
        output = model(batch_data[0].unsqueeze(0))
        assert output.shape == (1, 10), f"Output shape is {output.shape}, expected (1, 10)."

    # Test 5: Model inference timen for 1 batch of data
    start_time = time.time()
    with torch.no_grad():
        model(batch_data)
    inference_time = time.time() - start_time
    assert inference_time < 0.1, f"Inference time is {inference_time:.4f} seconds, expected < 0.1 seconds."

if __name__ == "__main__":
    test_model()