import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import torch.nn as nn
import torch.optim as optim

start_time = time.time()
print(f'Start time: {time.strftime("%H:%M:%S", time.localtime(start_time))}')
# Parameters
d = 5  # number of features (dimensions)
c = 5   # number of classes
n_per_class = 5  # samples per class
num_epochs = 100000000

# Generate class means, spaced apart
np.random.seed(0)
class_means = np.random.randn(c, d) * 5

def multiclass_functional_margin(W, X, y, reducer=np.array):
    W = W / np.linalg.norm(W)
    margins = []
    i_max_other_score_l = []
    for x, label in zip(X, y):
        scores = x@W  # shape (K,)
        true_score = scores[label]
        max_other_score = np.max(np.delete(scores, label))
        margins.append(true_score - max_other_score)
    return reducer(margins)

# Generate data
X_list = []
y_list = []
for i in range(c):
    X_class = np.random.randn(n_per_class, d) + class_means[i]
    y_class = np.full(n_per_class, i)
    X_list.append(X_class)
    y_list.append(y_class)

X = np.vstack(X_list)
y = np.concatenate(y_list)

# Shuffle the dataset
perm = np.random.permutation(len(X))
X = X[perm]
y = y[perm]

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Unique classes:", np.unique(y))


fig, axs = plt.subplots(1, 3, figsize=(15, 5))
# Define network parameters
for L in [1, 10]:
    # L = 3  # number of layers
    hidden_dim = d  # number of hidden units per layer

    # Convert data to torch tensors (double precision)
    X_tensor = torch.from_numpy(X).double()
    y_tensor = torch.from_numpy(y).long()

    # Define a deep linear network (no bias) in double precision
    class DeepLinearNet(nn.Module):
        def __init__(self, d, c, L):
            super().__init__()
            layers = []
            for i in range(L-1):
                layer = nn.Linear(d, d, bias=False)
                layer = layer.double()
                layers.append(layer)
            last_layer = nn.Linear(d, c, bias=False)
            last_layer = last_layer.double()
            layers.append(last_layer)
            self.layers = nn.ModuleList(layers)
        
        def forward(self, x):
            for layer in self.layers[:-1]:
                x = layer(x)
            x = self.layers[-1](x)
            return x

    model = DeepLinearNet(d, c, L)
    model = model.double()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    loss_l = []
    for epoch in range(num_epochs):
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_l.append(loss.item())
        if (epoch+1) % int(num_epochs/10) == 0 or epoch == 0:
            pred = outputs.argmax(dim=1)
            acc = (pred == y_tensor).float().mean().item()
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Acc: {acc*100:.2f}%")

    # Final accuracy
    with torch.no_grad():
        outputs = model(X_tensor)
        pred = outputs.argmax(dim=1)
        acc = (pred == y_tensor).float().mean().item()
        losses = torch.nn.functional.cross_entropy(outputs, y_tensor, reduction='none').numpy()
        print(f"Final training accuracy: {acc*100:.2f}%")

    W_prod = model(torch.eye(d).double()).detach().clone().numpy()
    W_norm = np.linalg.norm(W_prod)
    W_prod /= W_norm
    U, S, V = np.linalg.svd(W_prod)
    axs[0].plot(S, marker='o', label=f'L={L} --- W_norm={W_norm:.2f}')
    axs[1].plot(loss_l)
    axs[2].hist(multiclass_functional_margin(W_prod, X, y), alpha=0.8)
axs[0].legend()
axs[1].set_yscale('log')
axs[1].set_xscale('log')
axs[0].set_title('singular values')
axs[1].set_title('loss')
axs[2].set_title('functional margins')
plt.tight_layout()
plt.show()
end_time = time.time()
print(f'Start time: {time.strftime("%H:%M:%S", time.localtime(end_time))}')
print(f'Time taken: {(end_time - start_time)/60} minutes')