from matplotlib import pyplot as plt
import torch
from sklearn.decomposition import PCA
import numpy as np
from utils import calc_PR, compute_gradient, get_loss, get_state_dict_norm


def plot_loss_and_dist(data_dict):
    loc_y = data_dict['loc_y']; corridor = data_dict['corridor']; loss_l = data_dict['loss_l']; accuracy_l = data_dict['accuracy_l']
    X = data_dict['X']; y = data_dict['y'];  hidden_states = data_dict['hidden_states'];

    X_dist = torch.cdist(X, X).cpu().numpy()
    y_dist = torch.cdist(y, y).cpu().numpy()
    hidden_dist = torch.cdist(hidden_states[-1].detach(), hidden_states[-1].detach()).cpu().numpy()
    indices = np.lexsort((loc_y, corridor))
    # indices = indices[action_taken[indices]==0]
    fig, axs = plt.subplots(2, 3, figsize=(15/2, 10/2))
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


def plot_pca(data_dict, title="", axs=None):
    loc_y = data_dict['loc_y'];
    hidden_states = data_dict['hidden_states'];
    action_taken = data_dict['action_taken']
    loss_l = data_dict['loss_l']; accuracy_l = data_dict['accuracy_l']

    h_np = hidden_states[-1].cpu().detach().numpy()
    PR = calc_PR(h_np)

    pca = PCA().fit(h_np)
    X_reduced = pca.transform(h_np)
    if axs is None:
        fig, axs = plt.subplots(1, 4, figsize=(20/2, 5/2))
        axs[0].set_ylabel(title)
    # Add cumulative explained variance ratio in the first row
    ax1 = axs[0]
    ax1.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel('Cumulative EVR')
    ax1.set_title(f'Cumulative EVR --- PR: {PR:.2f}')
    ax1.set_ylim(-0.1, 1.1)

    ax1 = axs[1]
    s = ax1.scatter(X_reduced[:, 0], X_reduced[:, 1], c=loc_y, cmap='coolwarm', alpha=0.7)
    ax1.set_xlabel(f'Component 1')
    ax1.set_ylabel(f'Component 2'),
    ax1.axis('equal')

    ax1 = axs[2]
    ax1.scatter(X_reduced[:, 0], loc_y, c=loc_y)

    axs[3].plot(loss_l)
    axs[3].set_yscale('log')
    ax2 = axs[3].twinx()
    ax2.plot(accuracy_l, 'r')
    ax2.set_ylim(-0.1, 1.1)
    axs[3].set_title("Loss")

    plt.tight_layout()


def plot_solution_direction_loss_space(data_dict_l, labels_l, max_norm=50):
    norm_l = np.linspace(0.01, max_norm, 1000)
    plt.figure(figsize=(20, 5))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    for data_dict, label in zip(data_dict_l, labels_l):
        ax1.plot(norm_l, [get_loss(data_dict, normalize=norm) for norm in norm_l], label=label)
        ax2.plot(norm_l, [torch.norm(compute_gradient(data_dict, normalize=norm)).item() for norm in norm_l], label=label, ls='--', alpha=0.7)
        color = ax1.get_lines()[-1].get_color()  # get color of the last line plotted for this data_dict
        ax1.axvline(get_state_dict_norm(data_dict['initial_weights']), ls=':', alpha=0.7, color=color)
        ax1.axvline(get_state_dict_norm(data_dict['final_weights']), ls=':', alpha=1, color=color)
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax1.set_xlabel('Normalization Factor')
    ax1.set_ylabel('Loss')
    ax2.set_ylabel('Gradient Norm')
    # Set xticks every 2.5
    xticks = np.linspace(0, max_norm, 20)
    ax1.set_xticks(xticks)
    plt.legend()
    plt.show()

