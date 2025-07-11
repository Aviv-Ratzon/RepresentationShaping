import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Load the data from CSV file
data = pd.read_csv('data_weight.csv')
num = data['num'].values
weight_data = data['data'].values
num = num[~np.isnan(weight_data)]
weight_data = weight_data[~np.isnan(weight_data)]
# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(num, weight_data, 'b-', linewidth=1, marker='o')
plt.xlabel('num')
plt.ylabel('data')
plt.title('Data Weight Plot')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Convert data to PyTorch tensors
num_tensor = torch.tensor(num, dtype=torch.float32)
weight_data_tensor = torch.tensor(weight_data, dtype=torch.float32)

# Define the model class
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Initialize parameters as learnable tensors
        self.a = nn.Parameter(torch.tensor(-1.2/7, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(84.0, dtype=torch.float32))
        self.f = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.phi = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.A = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
    
    def forward(self, num):
        return self.a * num + self.b + self.A * torch.cos(2 * torch.pi * self.f * num + self.phi)

# Initialize model and optimizer
model = Model()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

# Training loop
num_epochs = 10000
losses = []

for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(num_tensor)
    loss = criterion(y_pred, weight_data_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.6f}')

# Extract optimized parameters
a_opt = model.a.item()
b_opt = model.b.item()
f_opt = model.f.item()
phi_opt = model.phi.item()
A_opt = model.A.item()

print(f"Optimized parameters:")
print(f"a = {a_opt:.6f}")
print(f"b = {b_opt:.6f}")
print(f"T = {1/f_opt:.6f}")
print(f"phi = {phi_opt:.6f}")
print(f"A = {A_opt:.6f}")
print(f"Final loss: {losses[-1]:.6f}")

# Generate predictions with optimized parameters
with torch.no_grad():
    y_pred = model(num_tensor).numpy()

# Plot the results
plt.figure(figsize=(12, 8))

# Original data
plt.subplot(2, 1, 1)
plt.plot(num, weight_data, 'b-', linewidth=1, marker='o', label='Original Data')
plt.plot(num, y_pred, 'r--', linewidth=2, label='Fitted Model')
plt.xlabel('num')
plt.ylabel('data')
plt.title('Data Fitting Results')
plt.legend()
plt.grid(True, alpha=0.3)

# Residuals
plt.subplot(2, 1, 2)
residuals = weight_data - y_pred
plt.plot(num, residuals, 'g-', linewidth=1, marker='o')
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.xlabel('num')
plt.ylabel('Residuals')
plt.title('Residuals (Original - Predicted)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Calculate R-squared
ss_res = np.sum(residuals**2)
ss_tot = np.sum((weight_data - np.mean(weight_data))**2)
r_squared = 1 - (ss_res / ss_tot)
print(f"R-squared: {r_squared:.6f}")
