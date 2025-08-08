import pickle as pkl
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
import multiprocessing as mp

start_time = time.time()

# Data: 3 points, 3 classes
X = np.array([
    [2, 1, 0],
    [0, 1, 0],
    [0, 1, 2]
], dtype=np.float32)
y = np.array([0, 1, 2], dtype=np.int64)

X_tensor = torch.tensor(X).double()
y_tensor = torch.tensor(y)

from utils import calc_PR

# Deep linear network: 3 -> 3 -> ... -> 3, all linear, no activation
class DeepLinearNet(nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(3, 3, bias=False) for _ in range(depth)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def run_depth(depth, n_epochs, lr_init, X_tensor, y_tensor, X, y):
    print(f"Running depth {depth}")
    start_time = time.time()
    # Each process gets its own lr, model, etc.
    lr = lr_init * (0.5 ** (depth-1))
    model = DeepLinearNet(depth).double()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_l = []
    for epoch in tqdm(range(n_epochs), desc=f"Depth {depth}", leave=False, disable=True):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        loss_l.append(loss.item())
    # Final predictions and metrics
    with torch.no_grad():
        outputs = model(X_tensor)
        pred = outputs.argmax(dim=1)
        acc = (pred == y_tensor).float().mean().item()
        final_loss = loss_l[-1]
    # Compute effective weight matrix
    W = model.layers[0].weight.detach().numpy().T
    for i in range(1, len(model.layers)):
        W = W @ model.layers[i].weight.detach().numpy().T
    W /= np.linalg.norm(W)
    scores = X @ W
    true_scores = scores[np.arange(3), y]
    scores_margin = scores.copy()
    scores_margin[np.arange(3), y] = -100
    margin = (true_scores[:, None] - scores_margin).min(1).min()
    pr = calc_PR(W)
    result = {
        "final_loss": final_loss,
        "accuracy": acc,
        "margin": margin,
        "PR": pr,
        "W": W,
        "pred": pred.numpy(),
        "loss_l": loss_l,
        "depth": depth
    }
    end_time = time.time()
    print(f"Depth {depth} took {(end_time - start_time)/60:.2f} minutes")
    return result

if __name__ == "__main__":
    depths = np.arange(1, 10)
    n_epochs = 1000#0000
    lr_init = 0.1
    results = {}

    # Use multiprocessing to run each depth in parallel
    with mp.Pool(processes=min(len(depths), mp.cpu_count())) as pool:
        # Prepare arguments for each process
        args = [(depth, n_epochs, lr_init, X_tensor, y_tensor, X, y) for depth in depths]
        results_list = list(pool.starmap(run_depth, args))

    # Collect results and plot loss curves
    for res in results_list:
        depth = res["depth"]
        results[depth] = {k: v for k, v in res.items() if k != "depth"}

    pkl.dump(results, open('test_multiclass_data_results.pkl', 'wb'))

    plt.figure(figsize=(10, 6))
    for depth in depths:
        plt.plot(results[depth]["loss_l"], label=f"Depth {depth}")
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss for Different Depths')
    plt.legend()
    plt.savefig('figures/loss_curves.png')
    plt.close()

    # Prepare data for plotting
    final_losses = [results[depth]['final_loss'] for depth in depths]
    accuracies = [results[depth]['accuracy'] for depth in depths]
    margins = [results[depth]['margin'] for depth in depths]
    prs = [results[depth]['PR'] for depth in depths]

    fig, axs = plt.subplots(2, 2, figsize=(6, 6))
    axs = axs.flatten()

    axs[0].plot(depths, final_losses, marker='o')
    axs[0].set_title('Final Loss vs Depth')
    axs[0].set_xlabel('Depth')
    axs[0].set_ylabel('Final Loss')
    axs[0].set_yscale('log')

    axs[1].plot(depths, accuracies, marker='o')
    axs[1].set_title('Accuracy vs Depth')
    axs[1].set_xlabel('Depth')
    axs[1].set_ylabel('Accuracy')

    axs[2].plot(depths, margins, marker='o')
    axs[2].set_title('Margin vs Depth')
    axs[2].set_xlabel('Depth')
    axs[2].set_ylabel('Margin')

    axs[3].plot(depths, prs, marker='o')
    axs[3].set_title('PR vs Depth')
    axs[3].set_xlabel('Depth')
    axs[3].set_ylabel('PR')

    plt.tight_layout()
    plt.savefig('figures/metrics_vs_depth.png')
    plt.close()

    # Optionally, also show predictions for each depth
    for depth in depths:
        r = results[depth]
        print(f"Depth {depth}: Predictions: {r['pred']}, True: {y}")

    end_time = time.time()
    print(f"Total run time: {(end_time - start_time)/60/60:.2f} hours")

    plt.scatter(margins, prs, c=depths, cmap='viridis')
    plt.colorbar()
    plt.savefig('figures/margin_vs_pr.png')
    plt.close()
