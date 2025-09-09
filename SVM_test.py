from utils import *
from data_modules import create_data
from run_sim import Config
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

alpha_l = []
X_l = []
W_l = []
y_l = []
action_taken_l = []
loc_y_l = []
loc_X_l = []
for max_move in [1, 5]:
    C = Config()

    C.length_corridors = [10]*1
    C.max_move = max_move

    X, y, corridor, loc_X, loc_y, action_taken, dim_l, input_size, output_size, n_actions = create_data(C)


    # Convert data to torch tensors
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y.argmax(1)).long()

    # If y is one-hot or images, try to get class labels
    if y_tensor.ndim > 1 and y_tensor.shape[1] > 1:
        # Try to infer class from y (e.g., MNIST images)
        # Here, we assume loc_y contains the class labels for y
        y_tensor = torch.from_numpy(loc_y).long()

    num_classes = len(torch.unique(y_tensor))
    input_dim = X_tensor.shape[1]

    class LinearSVM(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.fc = nn.Linear(input_dim, num_classes, bias=False)

        def forward(self, x):
            return self.fc(x)

    def multiclass_hinge_loss(outputs, targets, margin=1.0):
        # outputs: (batch, num_classes)
        # targets: (batch,)
        batch_size = outputs.size(0)
        correct_class_scores = outputs[torch.arange(batch_size), targets].unsqueeze(1)
        margins = outputs - correct_class_scores + margin
        margins[torch.arange(batch_size), targets] = 0  # Do not penalize correct class
        loss = torch.clamp(margins, min=0).sum() / batch_size
        return loss

    model = LinearSVM(input_dim, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 100000 * max_move

    loss_l = []
    for epoch in range(num_epochs):
        # No batching: train on all data at once
        outputs = model(X_tensor)
        loss = multiclass_hinge_loss(outputs, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_l.append(loss.item())
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    # Evaluate accuracy
    with torch.no_grad():
        outputs = model(X_tensor)
        preds = outputs.argmax(dim=1)
        acc = (preds == y_tensor).float().mean().item()
        print(f"Training accuracy: {acc*100:.2f}%")

    plt.plot(loss_l)
    plt.yscale('log')
    plt.xscale('log')
    plt.title(f'Accuracy: {acc*100:.2f}%')
    plt.show()

    W = model.fc.weight.detach().clone().numpy().T
    W_norm = W/np.linalg.norm(W)
    # U, S, V = np.linalg.svd(W_norm)
    # plt.plot(U[:,0])
    # plt.show()

    # plt.scatter(X@U[:,0]*S[0], X@U[:,1]*S[1], c=loc_y)
    # plt.axis('equal')
    # plt.show()

    import numpy as np

    def decompose_weights(X, W):
        """
        Decompose weight matrix W into linear combination coefficients alpha
        so that W[k] ≈ sum_i alpha[i,k] * X[i]
        
        X: (N, d) data matrix
        W: (K, d) weight matrix (one row per class)
        
        Returns:
            alpha: (N, K) coefficients for each point and class
        """
        N, d = X.shape
        K = W.shape[1]
        alpha = np.zeros((N, K))

        # Solve X^T alpha_k = w_k  --> alpha_k = argmin || X^T alpha_k - w_k ||
        # Equivalent: X^T (alpha_k) = w_k  --> alpha_k = least_squares(X, w_k)
        print(X.shape, W.shape)
        for k in range(K):
            # Solve min ||X^T alpha - w_k||^2  => alpha = least squares solution
            alpha[:, k], _, _, _ = np.linalg.lstsq(X.T, W[:,k], rcond=None)

        return alpha

    # Example usage
    # X: shape (N, d)
    # y: shape (N,)
    # W: shape (K, d)

    # alpha = decompose_weights(X, W)

    # # Verify reconstruction
    # W_reconstructed = X.T @ alpha # shape: (K, d)
    # print("Reconstruction error:", np.linalg.norm(W - W_reconstructed))
    # U, S, V = np.linalg.svd(W_reconstructed)
    # plt.plot(U[:,0])
    # plt.show()

    X_l.append(X)
    alpha_l.append(alpha)
    W_l.append(W)
    y_l.append(y)
    action_taken_l.append(action_taken)
    loc_y_l.append(loc_y)
    loc_X_l.append(loc_X)
# Compute the symmetric color scale limits across both alpha_l[0] and alpha_l[1]

stop

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
for alpha, ax in zip(alpha_l, axs.flatten()):
    vmax = max(np.abs(alpha).max(), np.abs(alpha).min())
    vmin = -vmax
    im0 = ax.imshow(alpha, cmap='berlin', vmin=vmin, vmax=vmax)
    plt.colorbar(im0, ax=ax)
plt.show()

import matplotlib.cm as cm
colors = cm.coolwarm(np.linspace(0, 1, alpha.shape[1]))
for alpha, action_taken, loc_y, loc_X in zip(alpha_l, action_taken_l, loc_y_l, loc_X_l):
    [plt.scatter(loc_X, alpha[:,i], c=colors[i]) for i in range(alpha.shape[1])]
    plt.xlabel('loc_X')
    plt.ylabel('alpha')
    plt.title('Weight of sample for each class by loc_X')
    plt.show()

for alpha, action_taken, loc_y, loc_X in zip(alpha_l, action_taken_l, loc_y_l, loc_X_l):
    [plt.scatter(action_taken, alpha[:,i], c=colors[i]) for i in range(alpha.shape[1])]
    plt.xlabel('action_taken')
    plt.ylabel('alpha')
    plt.title('Weight of sample for each class by action_taken')
    plt.show()

for alpha, X, W, y, action_taken, loc_X, loc_y in zip(alpha_l, X_l, W_l, y_l, action_taken_l, loc_X_l, loc_y_l):
    filter = abs(loc_y.squeeze()-4.5) > 2
    # Verify reconstruction
    W_reconstructed = X[filter].T @ alpha[filter] # shape: (K, d)
    print("Reconstruction error:", np.linalg.norm(W - W_reconstructed))
    U, S, V = np.linalg.svd(W_reconstructed)
    plt.plot(U[:,0])
    plt.show()

    
for alpha, action_taken, loc_y, loc_X in zip(alpha_l, action_taken_l, loc_y_l, loc_X_l):
    plt.plot(loc_y[alpha.argmax(0)])
plt.show()


for alpha, X, W, y, action_taken, loc_X, loc_y in zip(alpha_l, X_l, W_l, y_l, action_taken_l, loc_X_l, loc_y_l):
    U, S, V = np.linalg.svd(X)
    plt.plot(U[:,0])
    plt.show()

def multiclass_functional_margin(W, X, y, reducer=np.min, normalize=True):
    if normalize:
        W = W / np.linalg.norm(W)
    margins = []
    i_max_other_score_l = []
    for x, y_curr in zip(X, y):
        label = y_curr.argmax()
        scores = x@W  # shape (K,)
        true_score = scores[label]
        max_other_score = np.max(np.delete(scores, label))
        i_max_other_score = np.argmax(np.delete(scores, label))
        margins.append(true_score - max_other_score)
        i_max_other_score_l.append(i_max_other_score)
    return reducer(margins), np.argmin(margins), i_max_other_score_l[np.argmin(margins)]


max_move_l = [1, 5]
# Compute all margins first to determine common bins
all_margins = []
for X, y, W in zip(X_l, y_l, W_l):
    margins = multiclass_functional_margin(W, X, y, reducer=np.array, normalize=False)[0]
    all_margins.append(margins)
# Concatenate all margins to get global min/max
all_margins_concat = np.concatenate(all_margins)
# Choose bins (e.g., 20 bins between min and max)
n_bins = 20
bins = np.linspace(all_margins_concat.min(), all_margins_concat.max(), n_bins + 1)

fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
for ax, margins, max_move in zip(axs, all_margins, max_move_l):
    ax.hist(margins, bins=bins, alpha=1)
    ax.set_title(f'A={max_move}')
ax.set_xlabel('sample margins')
plt.show()

for W_norm in W_l:
    U, S, V = np.linalg.svd(W_norm)
    plt.plot(U[:,0])
    plt.show()

