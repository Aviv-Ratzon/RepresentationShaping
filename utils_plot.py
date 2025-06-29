from matplotlib import pyplot as plt
import torch
from sklearn.decomposition import PCA
import numpy as np


def plot_loss_and_dist(data_dict):
    loc_y = data_dict['loc_y']; corridor = data_dict['corridor']; loss_l = data_dict['loss_l']; accuracy_l = data_dict['accuracy_l']
    X = data_dict['X']; y = data_dict['y'];  hidden_states = data_dict['hidden_states'];

    X_dist = torch.cdist(X, X).cpu().numpy()
    y_dist = torch.cdist(y, y).cpu().numpy()
    hidden_dist = torch.cdist(hidden_states[-1].detach(), hidden_states[-1].detach()).cpu().numpy()
    indices = np.lexsort((loc_y, corridor))
    # indices = indices[action_taken[indices]==0]
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs[0, 0].set_axis_off();
    axs[0, 2].set_axis_off()
    axs[0, 1].plot(loss_l)
    axs[0, 1].set_yscale('log')
    axs[0, 1].twinx().plot(accuracy_l, 'r')
    axs[0, 1].set_title("Loss")
    for var, var_name, ax in zip([X_dist, y_dist, hidden_dist], ['X', 'y', 'hidden'], axs[1]):
        ax.imshow(var[indices][:, indices], cmap='viridis')
        ax.set_title(f'{var_name} distance matrix')
        ax.grid(False)
    plt.tight_layout()
    plt.show()


def plot_pca(data_dict):
    loc_y = data_dict['loc_y'];
    hidden_states = data_dict['hidden_states'];
    action_taken = data_dict['action_taken']

    h_np = hidden_states[-1].cpu().detach().numpy()

    pca = PCA().fit(h_np)
    X_reduced = pca.transform(h_np)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # Add cumulative explained variance ratio in the first row
    ax1 = axs[0]
    ax1.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel('Cumulative Explained Variance')
    ax1.set_title(f'Cumulative Explained Variance Ratio')

    ax1 = axs[1]
    s = ax1.scatter(X_reduced[:, 0], X_reduced[:, 1], c=loc_y, cmap='coolwarm', alpha=0.7)
    ax1.set_xlabel(f'Component 1')
    ax1.set_ylabel(f'Component 2'),
    ax1.axis('equal')

    ax1 = axs[2]
    ax1.scatter(X_reduced[:, 0], loc_y, c=loc_y)

    plt.tight_layout()
    plt.show()