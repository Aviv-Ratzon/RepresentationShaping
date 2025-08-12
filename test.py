import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Parameters
L = 10  # Number of hidden layers
N = 64  # Number of neurons per layer
input_size = 3
output_size = 2
num_epochs = 10000
batch_size = 2
learning_rate = 0.1
input_noise = 0
# Generate dataset: [1,0,0] -> [1,0], [0,1,0] -> [0,1]
X = np.array([[1,0,0], [0,1,0]])
y = np.array([[1,0], [0,1]])

# Repeat to make a larger dataset
# X = np.repeat(X, 100, axis=0)
# y = np.repeat(y, 100, axis=0)

# Shuffle
idx = np.random.permutation(len(X))
X = X[idx]
y = y[idx]

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Define the DNN
class SimpleDNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        for _ in range(num_layers-1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_size))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

model = SimpleDNN(input_size, N, output_size, L)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

loss_l = []
# Training loop
for epoch in tqdm(range(num_epochs)):
    permutation = torch.randperm(X_tensor.size(0))
    for i in range(0, X_tensor.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_tensor[indices], y_tensor[indices]
        outputs = model(batch_x + torch.randn_like(batch_x)*input_noise)
        # CrossEntropyLoss expects class indices, not one-hot

        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_l.append(loss.item())
    if (epoch+1) % (num_epochs//10) == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
plt.plot(loss_l)
plt.yscale('log')
plt.show()
# Test
with torch.no_grad():
    test_X = torch.tensor([[1,0,0],[0,1,0]], dtype=torch.float32)
    test_out = model(test_X)
    print("Test predictions:", test_out.tolist())  # Should be [0,1]

X_test = torch.tensor([[1,0.5,0],[0.5,1,0]], dtype=torch.float32)
y_test = torch.tensor([[1,0],[0,1]], dtype=torch.float32)
test_out = model(X_test)
test_loss = criterion(test_out, y_test)
print(test_loss)







