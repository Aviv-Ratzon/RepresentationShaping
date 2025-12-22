import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for multiprocessing
from matplotlib import pyplot as plt

import torch
import numpy as np
from copy import deepcopy
import multiprocessing as mp
from multiprocessing import Pool
import os

from torch import nn
from umap import UMAP

from tqdm import tqdm
from utils import *
import matplotlib as mpl
import scipy
from scipy.special import softmax
from functools import reduce
import itertools

import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class LinearRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        # Recurrent parameters
        self.W_ih = nn.ParameterList()
        self.W_hh = nn.ParameterList()
        self.bias = nn.ParameterList()

        for l in range(n_layers):
            in_size = input_size if l == 0 else hidden_size
            self.W_ih.append(nn.Parameter(torch.randn(hidden_size, in_size) * 0.1))
            self.W_hh.append(nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1))
            self.bias.append(nn.Parameter(torch.zeros(hidden_size)))

        # Linear readout applied on the last layer
        self.readout = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x, h0=None):
        """
        x:  (batch, time, input_size)
        h0: (n_layers, batch, hidden_size), optional
        
        Returns:
            y:    (batch, time, output_size)     -> readout from last layer
            h_T:  (batch, time, hidden_size)     -> last layer activations
        """

        B, T, _ = x.shape

        if h0 is None:
            h = torch.zeros(self.n_layers, B, self.hidden_size, device=x.device)
        else:
            h = h0

        last_layer_acts = []
        readouts = []

        for t in range(T):
            inp = x[:, t]  # (B, input_size)
            new_h = []

            for l in range(self.n_layers):
                prev_h = h[l]  # (B, hidden_size)

                next_h = (
                    inp @ self.W_ih[l].T +
                    prev_h @ self.W_hh[l].T +
                    self.bias[l]
                )  # (B, hidden_size)

                new_h.append(next_h)
                inp = next_h  # feed to next layer

            h = torch.stack(new_h, dim=0)

            # top-layer activation (B, hidden_size)
            h_last = h[-1]
            last_layer_acts.append(h_last)

            # readout (B, output_size)
            y_t = self.readout(h_last)
            readouts.append(y_t)

        # Stack over time
        last_layer_acts = torch.stack(last_layer_acts, dim=1)  # (B, T, hidden_size)
        readouts = torch.stack(readouts, dim=1)                # (B, T, output_size)

        return readouts, last_layer_acts.detach().cpu().numpy()



# Standard PyTorch RNN module with L layers
class StandardRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(StandardRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.rnn = LinearRNN(input_size, hidden_size, num_layers)
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,      # input (N, T, F)
            nonlinearity='tanh'
        )
        self.fc = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        batch_size = x.size(0)
        device = x.device
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        out_rnn, h_n = self.rnn(x, h0)                  # out_rnn: (N, T, hidden)
        outputs = self.fc(out_rnn)                       # (N, T, output_size)
        # Optionally collect all hidden states for visualization (reshape for compatibility)
        h = out_rnn.detach().cpu().numpy()             # (N, T, hidden_size)
        return outputs, h


def train_and_plot(config):
    """Train model and generate plot for given A, num_layers, and loss_fn"""
    # A, num_layers, loss_fn_name, gpu_id = args
    A = config['A']
    net_type = config['net_type']
    num_layers = config['num_layers']
    loss_fn = config['loss_fn']
    gpu_id = config['gpu_id']
    data_type = config['data_type']
    
    # Set device for this process
    if torch.cuda.is_available() and torch.cuda.device_count() > gpu_id:
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(gpu_id)
    else:
        device = torch.device('cpu')
    
    S = 40
    X = []
    y = []
    actions_in = [[1,0,0], [0,1,0], [0,0,1]]
    actions = [-1, 0, 1]
    states_obs = np.eye(S)

    # Generate data based on A, all lengths
    # for s in range(S):
    #     for a, a_in in zip(actions, actions_in):
    #         if a == 0:
    #             continue
    #         for n_actions in range(1, A+1):
    #             s_curr = s
    #             a_curr = a
    #             a_in_curr = a_in
    #             X_curr = []
    #             y_curr = []
    #             for action_num in range(A):
    #                 if action_num > n_actions or (s_curr + a_curr) < 0 or (s_curr + a_curr) >= S:
    #                     a_curr = 0
    #                     a_in_curr = actions_in[1]
    #                 X_curr.append(np.concatenate([states_obs[s_curr]*int(action_num==0), a_in_curr]))
    #                 s_curr = (s_curr + a_curr) % S
    #                 y_curr.append(states_obs[s_curr])
    #             X.append(X_curr)
                # y.append(y_curr)

    # Generate data based on A, only length A
    for s in range(S):
        for a, a_in in zip(actions, actions_in):
            n_actions = A
            s_curr = s
            a_curr = a
            a_in_curr = a_in
            X_curr = []
            y_curr = []
            for action_num in range(A):
                if (((s_curr + a_curr) < 0) or ((s_curr + a_curr) >= S)) & ('linear' in data_type):
                    a_curr = 0
                    a_in_curr = actions_in[1]
                X_curr.append(np.concatenate([states_obs[s_curr]*int(action_num==0), a_in_curr]))
                s_curr = (s_curr + a_curr) % S
                y_curr.append(states_obs[s_curr])
            X.append(X_curr)
            y.append(y_curr)

    X = np.array(X)
    y = np.array(y)

    # Prepare data for PyTorch: (N, T, F)
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)  # shape: (N, T, F)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)  # shape: (N, T, F_s)

    N, T, F = X_tensor.shape
    _, _, F_s = y_tensor.shape

    # For classification (one-hot to class index)
    y_class = torch.argmax(y_tensor, dim=2).to(device)  # shape: (N, T)

    hidden_size = 128
    if net_type == 'nonlinear':
        model = StandardRNN(X.shape[2], hidden_size, y.shape[2], num_layers=num_layers).to(device)
    elif net_type == 'linear':
        model = LinearRNN(X.shape[2], hidden_size, y.shape[2], n_layers=num_layers).to(device)
    else:
        raise ValueError(f'Invalid net_type: {net_type}')

    # Create loss function based on loss_fn_name
    criterion = loss_fn
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    n_epochs = 50000

    loss_list = []
    acc_list = []

    # Training loop (without progress bar for cleaner multiprocessing output)
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        output, h_l = model(X_tensor)  # (N, T, F_s)
        # CrossEntropyLoss expects (N, C) and (N) for 2d case, so flatten batch and seq
        # MSELoss expects (N, C) and (N, C) - both need same shape (one-hot)
        # CrossEntropyLoss expects (N, C) and (N) - target is class indices

        output_flat = output.view(-1, F_s)
        target_flat = y_tensor.view(-1, F_s) if isinstance(criterion, nn.MSELoss) else y_class.view(-1)
        loss = criterion(output_flat, target_flat)
        loss.backward()
        optimizer.step()
        # Calculate accuracy
        with torch.no_grad():
            preds = torch.argmax(output, dim=2)  # (N, T)
            correct = (preds == y_class).float().sum()
            total = preds.numel()
            accuracy = correct / total
        # Track for plotting
        loss_list.append(loss.item())
        acc_list.append(accuracy.item())

    hidden = np.array(h_l)
    hidden_reshaped = hidden.reshape(-1, hidden.shape[2]).T  # shape: (neurons, time*batch)

    # Perform PCA
    n_components = min(30, hidden_reshaped.shape[1])
    pca_full = PCA(n_components=n_components)
    hidden_pca_all = pca_full.fit_transform(hidden_reshaped.T)
    explained_variance_ratio = np.cumsum(pca_full.explained_variance_ratio_)

    # 2D PCA for plotting
    pca = PCA(n_components=2)
    hidden_pca = pca.fit_transform(hidden_reshaped.T)
    target_pos = y.reshape(-1, S).argmax(axis=1)
    curr_pos = X.reshape(-1, S+3)[:,:-3].argmax(axis=1)
    curr_action = X.reshape(-1, S+3)[:,-3:].argmax(axis=1)

    # Compute distance matrix of hidden states (cosine for example)
    dists = squareform(pdist(hidden_reshaped.T, metric='cosine'))

    # Ensure the output directory exists
    sub_path = '/'.join(['_'.join([s1, str(s2)]) for s1, s2 in zip(list(config.keys())[:-2], list(config.values())[:-2])])
    output_dir = f"figures/abstract_rnn/{sub_path}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"A_{A}.png")

    # Plot all four subplots
    fig, axes_all = plt.subplots(2, 3, figsize=(15,10))

    axes = axes_all[0]
    # 1. Cumulative Explained Variance
    axes[0].plot(np.arange(1, n_components+1), explained_variance_ratio, marker='o')
    axes[0].set_xlabel('Number of Principal Components')
    axes[0].set_ylabel('Cumulative Explained Variance')
    axes[0].set_title('PCA Variance Explained')
    axes[0].set_ylim([0, 1.01])

    # 3. Loss and accuracy over training (right y-axis for acc)
    ax3 = axes[1]
    ax3.plot(loss_list, color='tab:blue', label='Loss')
    ax3.set_ylabel('Loss', color='tab:blue')
    ax3.set_xlabel('Epoch')
    ax3.set_yscale('log')
    ax3.set_title('Training Loss and Accuracy')
    ax4 = ax3.twinx()
    ax4.plot(acc_list, color='tab:red', label='Accuracy')
    ax4.set_ylabel('Accuracy', color='tab:red')
    ax4.set_ylim([0, 1.01])
    ax3.legend(loc='upper right')
    ax4.legend(loc='lower right')

    # 4. Distance matrix
    # Sort hidden states and dists by target_pos
    sort_idx = np.argsort(target_pos)
    dists_sorted = dists[sort_idx][:, sort_idx]
    im = axes[2].imshow(dists_sorted, aspect='auto', cmap='viridis')
    axes[2].set_title('Distance Matrix (cosine, sorted)')
    axes[2].set_xlabel('Sorted Time*Batch Index')
    axes[2].set_ylabel('Sorted Time*Batch Index')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    
    axes = axes_all[1]
    # 2. 2D PCA projection
    ax = axes[0]
    sc = ax.scatter(hidden_pca[:, 0], hidden_pca[:, 1], c=target_pos, cmap='coolwarm', alpha=0.6)
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_title('2D PCA Projection of Hidden States')
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label='Next State')
    
    ax = axes[1]
    sc = ax.scatter(hidden_pca[:, 0], hidden_pca[:, 1], c=curr_pos, cmap='coolwarm', alpha=0.6)
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_title('2D PCA Projection of Hidden States')
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label='Current State')
    
    ax = axes[2]
    sc = ax.scatter(hidden_pca[:, 0], hidden_pca[:, 1], c=curr_action, cmap='coolwarm', alpha=0.6)
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_title('2D PCA Projection of Hidden States')
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label='Current Action')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    
    return (A, num_layers, loss_fn, gpu_id)


if __name__ == '__main__':
    # Get number of available GPUs
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f'Found {num_gpus} GPU(s)')
    else:
        num_gpus = 1
        print('No GPUs found, using CPU')

    # Sweep parameters - get all dicts with unique value combinations of keys
    config_dict = {
        'net_type': ['linear', 'nonlinear'],
        'data_type': ['linear', 'cyclic'],
        'loss_fn': [nn.CrossEntropyLoss(), nn.MSELoss()],
        'num_layers':  [1, 5],
        'A': list(range(1, 20)),
    }

    # Get list of all dicts with all key-value combinations
    keys, values = zip(*config_dict.items())
    all_configs = [dict(zip(keys, v)) for v in product(*values)]
    all_configs = [{**dict, 'gpu_id': i % num_gpus} for i, dict in enumerate(all_configs)]
    # A_values = list(range(1, 20))  # [1, 15]
    # num_layers_values = [1, 5]  # [1, 5]
    # loss_fn_names = ['CrossEntropyLoss', 'MSELoss']
    
    # Create all combinations of (A, num_layers, loss_fn_name)
    # param_combinations = list(itertools.product(A_values, num_layers_values, loss_fn_names))
    
    # Assign GPU IDs in round-robin fashion
    # tasks = [(A, num_layers, loss_fn_name, i % num_gpus) for i, (A, num_layers, loss_fn_name) in enumerate(param_combinations)]
    
    print(f'Total jobs: {len(all_configs)}')
    print(f'Running on {num_gpus} GPU(s)')
    
    # Use multiprocessing Pool
    # Use spawn method for better GPU isolation
    # Use more processes than GPUs to allow parallel jobs per GPU (if memory allows)
    num_processes = min(len(all_configs), num_gpus * 2) if torch.cuda.is_available() else mp.cpu_count()
    print(f'Using {num_processes} processes')
    
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Start method already set
        pass
    
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(train_and_plot, all_configs),
            total=len(all_configs),
            desc="Processing jobs"
        ))
    
    print('All jobs completed!')

