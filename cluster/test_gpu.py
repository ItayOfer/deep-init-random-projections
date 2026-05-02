import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("ERROR: GPU is not accessible. Slurm might not have allocated one, or the container lacks NVIDIA drivers.")
    sys.exit(1)

print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
print(f"GPU Count: {torch.cuda.device_count()}")

# 1. Define a simple dummy CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 28 * 28, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        return self.fc(x)

# 2. Move model to GPU
device = torch.device("cuda")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Create dummy data (simulating a batch of 28x28 grayscale images)
inputs = torch.randn(64, 1, 28, 28).to(device)
targets = torch.randint(0, 10, (64,)).to(device)

print("\nStarting training loop on dummy data...")
start_time = time.time()
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/10 - Loss: {loss.item():.4f}")

print(f"\nTraining completed in {time.time() - start_time:.2f} seconds.")
print("SUCCESS: The GPU is fully accessible and functioning perfectly!")