import pickle as pkl
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
import multiprocessing as mp

def calc_margins(W, X, y):
    factor = 1/np.linalg.norm(W)**2
    W = np.sqrt(factor) * W
    margins = []
    for i in range(len(X)):
        true_class = y[i]
        x_i = X[i]
        true_score = np.dot(W[:, true_class], x_i)
        other_scores = [np.dot(W[:, j], x_i) for j in range(W.shape[0]) if j != true_class]
        margin = true_score - max(other_scores)
        margins.append(margin)
    return margins

def get_effective_W(model):
    depth = len(model.layers)
    W = model.layers[0].weight.T
    for d in range(1, depth, 1):
        W = W @ model.layers[d].weight.T
    W = W.detach().cpu().numpy()
    return W

start_time = time.time()
num_epochs = 10000000
lr = 0.01
depths = range(1, 6)
# Data: 3 points, 3 classes
X = np.array([
    [2, 1, 0],
    [0, 1, 0],
    [0, 1, 2]
], dtype=np.float32)
X = np.array([
    [2, 1, 0],
    [0.9, 1, 0],
    [0, 1, 0],
    [0, 1, 2],
    [1.1, 1, 2]
], dtype=np.float32)
y = np.array([0, 1, 2], dtype=np.int64)
y = np.array([0, 0, 1, 2, 2], dtype=np.int64)

from sklearn.svm import LinearSVC

# Train a linear multiclass SVM
svm = LinearSVC(C=1e6, multi_class='ovr', fit_intercept=False, max_iter=10000)
svm.fit(X, y)

# Compute functional margins for each sample
W = svm.coef_  # shape (n_classes, n_features)
scores = X@W
scores[np.arange(len(y)), y] = 0
print(f'SVM scores positive to wrong class: {(scores>0).any()}')


svm_margins = min(calc_margins(svm.coef_, X, y))
print("SVM functional margins:", svm_margins)

X_tensor = torch.tensor(X).double()
y_tensor = torch.tensor(y)

# Deep linear network: 3 -> 3 -> ... -> 3, all linear, no activation
class DeepLinearNet(nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(3, 3, bias=False) for _ in range(depth)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

        # INSERT_YOUR_CODE
import torch
from torch import nn, optim

margins_by_depth = []
loss_l_l = []
margins_l_l = []
models_l = []
for depth in depths:
    model = DeepLinearNet(depth).double()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    # Train for a fixed number of epochs
    loss_l = []
    margins_l = []
    for epoch in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        loss_l.append(loss.item())
        margins_l.append(min(calc_margins(get_effective_W(model), X, y)))
    # After training, compute functional margins
    with torch.no_grad():
        W = get_effective_W(model)
        margins = calc_margins(W, X, y)
        margins_by_depth.append(min(margins))
        loss_l_l.append(loss_l)
        margins_l_l.append(margins_l)
        models_l.append(model)

pkl.dump(models_l, open('results/models_test_multiclass.pkl', 'wb'))

models_l = pkl.load(open('results/models_test_multiclass.pkl', 'rb'))
margins_l = []
W_l = []
for model in models_l:
    W = get_effective_W(model)
    W_l.append(W)
    margins_l.append(calc_margins(W, X, y))
plt.boxplot(margins_l)
plt.show()
# INSERT_YOUR_CODE

# Create a similarity matrix of all matrices in W_l invariant to norm
# We'll use cosine similarity between flattened matrices (invariant to norm)
import numpy as np

num_models = len(W_l)
similarity_matrix = np.zeros((num_models, num_models))

# Flatten and normalize each W
W_flat_normed = []
for W in W_l:
    w_flat = W.flatten()
    w_flat = w_flat / np.linalg.norm(w_flat)
    W_flat_normed.append(w_flat)

for i in range(num_models):
    for j in range(num_models):
        similarity_matrix[i, j] = np.dot(W_flat_normed[i], W_flat_normed[j])

plt.figure(figsize=(6,5))
plt.imshow(similarity_matrix, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Cosine Similarity')
plt.title('Cosine Similarity Matrix of Effective Weights (Norm-Invariant)')
plt.xlabel('Model Index')
plt.ylabel('Model Index')
plt.tight_layout()
plt.show()


plt.plot(np.array(loss_l_l).T)
plt.yscale('log')
plt.show()

plt.plot(np.array(margins_l_l).T)
plt.ylim(0, svm_margins*1.1)
plt.show()

# Plot
plt.figure(figsize=(6,4))
plt.plot(depths, margins_by_depth, marker='o', label='Deep Linear Net')
# plt.axhline(y=svm_margins, color='r', linestyle='--', label='SVM')
plt.xlabel("Network Depth")
plt.ylabel("Minimal Functional Margin")
plt.title("Functional Margins vs. Depth (Deep Linear Net)")
plt.legend()
plt.tight_layout()
plt.show()
