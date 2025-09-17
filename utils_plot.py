from matplotlib import pyplot as plt
import torch
from sklearn.decomposition import PCA
import numpy as np
from utils import *


markers = ['o', 'x', '*', 'v', '^', 'p', 'h', '8', 'X', 'd']

def plot_loss_and_dist(data_dict):
    loc_y = data_dict['loc_y']; corridor = data_dict['corridor']; loss_l = data_dict['loss_l']; accuracy_l = data_dict['accuracy_l']
    X = data_dict['X']; y = data_dict['y'];  hidden_states = data_dict['hidden_states'];

    X_dist = torch.cdist(X, X).cpu().numpy()
    y_dist = torch.cdist(y, y).cpu().numpy()
    hidden_dist = torch.cdist(hidden_states[-1].detach(), hidden_states[-1].detach()).cpu().numpy()
    indices = y.argmax(1).argsort()
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
    loc_y = data_dict['loc_y']
    hidden_states = data_dict['hidden_states']
    action_taken = data_dict['action_taken']
    loss_l = data_dict['loss_l']; accuracy_l = data_dict['accuracy_l']
    y = data_dict['y'].cpu().numpy()
    C = data_dict['C']
    corridor = data_dict['corridor']
    X_np = data_dict['X'].cpu().numpy()
    final_weights = data_dict['final_weights']

    hidden = hidden_states[-1].cpu().detach().numpy()
    hidden_dist = torch.cdist(hidden_states[-1], hidden_states[-1]).cpu().numpy()

    color = loc_y if C.corridor_dim == 1 else loc_y[:, 0]

    PR = calc_PR(hidden)
    if not C.bias:
        W_effective = get_effective_W_from_model_dict(final_weights).cpu().numpy()
        W_PR = calc_PR(W_effective)
        if C.L == 0:
            U, S, V = np.linalg.svd(W_effective, full_matrices=False)
            hidden = X_np @ U @ np.diag(S)
            hidden_dist = torch.cdist(torch.tensor(hidden), torch.tensor(hidden)).cpu().numpy()
    else:
        W_PR = 0
    hidden_pr = calc_PR(hidden)
    
    # alignment = alignment_score(hidden[corridor==0], hidden[corridor==1]) if n_corridors > 1 else 0
    order = get_r_2(PCA(n_components=C.corridor_dim).fit_transform(hidden), loc_y)

    pca = PCA().fit(hidden)
    X_reduced = pca.transform(hidden)
    if axs is None:
        fig, axs = plt.subplots(1, 5, figsize=(25/2, 5/2))
        axs[0].set_ylabel(title)
    # Add cumulative explained variance ratio in the first row
    ax1 = axs[0]
    ax1.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    ax1.set_xlabel('Number of Components')
    # ax1.set_ylabel('Cumulative EVR')
    ax1.set_title(f'order = {order:.2f} --- W PR = {W_PR:.2f} --- hidden PR = {hidden_pr:.2f}')
    ax1.set_ylim(-0.1, 1.1)
    
    ax1 = axs[1]
    for cor, marker in zip(np.unique(corridor), markers):
        ax1.scatter(X_reduced[corridor==cor, 0], X_reduced[corridor==cor, 1], c=color[corridor==cor], cmap='coolwarm', alpha=0.7, marker=marker)
    ax1.set_xlabel(f'Component 1')
    ax1.set_ylabel(f'Component 2'),
    ax1.axis('equal')

    ax1 = axs[2]
    ax1.scatter(X_reduced[:, 0], color, c=color)

    axs[3].plot(loss_l)
    axs[3].set_yscale('log')
    ax2 = axs[3].twinx()
    ax2.plot(accuracy_l, 'r')
    ax2.set_ylim(-0.1, 1.1)
    axs[3].set_title("Loss")

    axs[4].imshow(hidden_dist[y.argmax(1).argsort()][:, y.argmax(1).argsort()], cmap='viridis')
    axs[4].set_title('hidden distance matrix')
    axs[4].grid(False)

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


def plot_pca_1d(data_dict, title="", axs=None):
    loc_y = data_dict['loc_y']
    hidden_states = data_dict['hidden_states']
    action_taken = data_dict['action_taken']
    loss_l = data_dict['loss_l']; accuracy_l = data_dict['accuracy_l']
    C = data_dict['C']
    X_np = data_dict['X'].cpu().numpy()
    final_weights = data_dict['final_weights']

    hidden = hidden_states[-1].cpu().detach().numpy()

    if C.L == 0:
        W_effective = get_effective_W_from_model_dict(final_weights).cpu().numpy()
        U, S, V = np.linalg.svd(W_effective, full_matrices=False)
        hidden = X_np @ U @ np.diag(S)
    
    # alignment = alignment_score(hidden[corridor==0], hidden[corridor==1]) if n_corridors > 1 else 0
    order = get_r_2(PCA(n_components=1).fit_transform(hidden), loc_y)

    pca = PCA().fit(hidden)
    X_reduced = pca.transform(hidden)
    if axs is None:
        fig, axs = plt.subplots(1, 4, figsize=(20/2, 5/2))
        axs[0].set_ylabel(title)
    # Add cumulative explained variance ratio in the first row
    ax1 = axs[0]
    ax1.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    ax1.set_xlabel('Number of Components')
    # ax1.set_ylabel('Cumulative EVR')
    ax1.set_title(f'order = {order:.2f}')
    ax1.set_ylim(-0.1, 1.1)

    ax1 = axs[1]
    s = ax1.scatter(X_reduced[:, 0], np.zeros_like(X_reduced[:, 0]), c=loc_y, cmap='coolwarm', alpha=0.7)
    ax1.set_xlabel(f'Component 1')
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

    

def plot_pca_subplot(ax, data_dict, title, cb=False):
    h = data_dict['hidden_states'][-1].cpu().numpy()
    loc_y = data_dict['loc_y']
    action_taken = data_dict['action_taken']
    corridor = data_dict['corridor']
    cond = abs(action_taken) <= 1
    # If loc_y is 2D, color by the first dimension
    color = loc_y[:, 0] if loc_y.ndim > 1 else loc_y
    color = color[cond]
    h = h[cond]
    corridor = corridor[cond]
    # PCA to 2D
    from sklearn.decomposition import PCA
    h_pca = PCA(n_components=2).fit_transform(h)
    
    # Get corridor information and markers
    markers = ['o', 'v', '*', 'v', '^', 'p', 'h', '8', 'X', 'd']
    
    # Plot each corridor with different markers
    for cor, marker in zip(np.unique(corridor), markers):
        mask = corridor == cor
        sc = ax.scatter(
            h_pca[mask, 0], h_pca[mask, 1], c=color[mask], cmap='viridis',
            s=80, alpha=0.9, edgecolor='none', marker=marker
        )
    ax.set_title(title, fontsize=20, pad=10)
    ax.axis('equal')
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)
    if cb:
        cbar = plt.colorbar(sc, ax=ax, pad=0.01, fraction=0.05)
        cbar.ax.set_yticklabels([])  # Remove colorbar ticks
        cbar.set_label('location', fontsize=16)