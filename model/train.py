import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from net import Net
from tqdm import tqdm
from eval import train,  test

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST dataset
train_transforms = transforms.Compose([
    transforms.RandomApply([transforms.CenterCrop(32), ], p=0.1),
    transforms.Resize((28, 28)),
    transforms.RandomRotation((-5., 5.), fill=0),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ])

# Test data transformations
test_transforms = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ])

train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
test_data = datasets.MNIST('../data', train=True, download=True, transform=test_transforms)

batch_size = 2

kwargs = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 2, 'pin_memory': True}

test_loader = torch.utils.data.DataLoader(test_data, **kwargs)
train_loader = torch.utils.data.DataLoader(train_data, **kwargs)

# Initialize the model, loss function, and optimizer
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.00075, momentum=0.9) # Initialize the optimizer to SGD with learning rate 0.01 and momentum 0.9.
num_epochs = 1

for epoch in range(1, num_epochs+1):
  print(f'Epoch {epoch}')
  train(model, device, train_loader, optimizer, criterion)
  test(model, device, train_loader, criterion)

# Save the model in the specified path
torch.save(model.state_dict(), 'model/mnist_model.pth')