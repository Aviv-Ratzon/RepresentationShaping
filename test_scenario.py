import pickle
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, gaussian_filter
from sklearn.preprocessing import OneHotEncoder

# n_classes = 30
# n_features = 100
# samples_per_class = 100
# sigma = 2/(10*n_classes)
# frac_overlap = 0.9

# X_l = []
# for frac in [0, frac_overlap]:
#     X = np.concatenate([np.random.randn(samples_per_class, n_features)*sigma + 1 for i in range(n_classes)],0)
#     X = gaussian_filter(X, sigma=[100,0])
#     X /= X.std(axis=0)
#     y = np.repeat(np.arange(n_classes), samples_per_class)
#     for n_class in range(n_classes):
#         D = int(n_features / ((1-frac)*n_classes))
#         mask = np.eye(n_features)[(n_class+1)*n_features//(n_classes+1)]
#         mask = convolve(mask, np.ones(D)) < 1
#         for i in np.where(y==n_class)[0]:
#             for j in np.where(mask)[0]:
#                     X[i][j] = 0
#     X_l.append(X)
# X = np.concatenate(X_l, axis=1)
with open('Sigma.pkl', 'rb') as f:
    Sigma = pickle.load(f)
data = np.random.multivariate_normal(np.zeros(Sigma.shape[0]), Sigma, 100000)
X = data[:,:61]
y = data[:,61:]


# import matplotlib.pyplot as plt
# plt.scatter(X[:,0], X[:,1], c=y)
# plt.show()
# Standardize the data
# Convert data to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)  # Changed to long type for CrossEntropyLoss

# plt.imshow(X[::X.shape[0]//n_features])
# plt.show()

# Define a deep linear network
class DeepLinearNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=4):
        super(DeepLinearNet, self).__init__()
        layers = [nn.Linear(input_size, hidden_size, bias=False)]
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size, bias=False))
        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x):
        hidden = self.layers(x)
        return self.output_layer(hidden), hidden


# Model, loss function, and optimizer
input_size = X.shape[1]
hidden_size = 250
output_size = n_classes
model = DeepLinearNet(input_size, hidden_size, output_size, 10)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 100
losses = []
accuracies = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs, _ = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    
    # Record loss
    losses.append(loss.item())
    
    # Record accuracy
    with torch.no_grad():
        predictions = torch.argmax(outputs, dim=1)
        # accuracy = (predictions == y).float().mean()
        # accuracies.append(accuracy.item())

    if (epoch + 1) % (num_epochs//10) == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Plot training curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(accuracies)
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.show()

# Evaluate the model
model.eval()
with torch.no_grad():
    outputs, hidden = model(X)
    predictions = torch.argmax(outputs, dim=1)
    accuracy = (predictions == y).float().mean()
    print(f'Accuracy: {accuracy.item():.4f}')


# Perform PCA on hidden activations and input data
pca_hidden = PCA(n_components=2)
pca_input = PCA(n_components=2)

hidden_np = hidden.detach().numpy()
X_np = X.detach().numpy()

hidden_pca = pca_hidden.fit_transform(hidden_np)
X_pca = pca_input.fit_transform(X_np)

# Create scatter plots
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA of Input Data')

plt.subplot(1, 2, 2) 
plt.scatter(hidden_pca[:, 0], hidden_pca[:, 1], alpha=0.5)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA of Hidden Layer Activations')

plt.tight_layout()
plt.show()