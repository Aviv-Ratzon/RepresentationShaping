import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

# Set global matplotlib formatting for ICLR paper compatibility
# Set global font to Times New Roman
# import seaborn as sns
# sns.set_context("paper")
# sns.set_style("whitegrid")
FIG_WIDTH = 5.5

mpl.rcParams.update({
    'font.size': 15,
    'axes.labelsize': 8,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 5,
    'legend.fontsize': 8,
    'figure.titlesize': 30,
    # 'axes.linewidth': 1.2,
    # 'lines.linewidth': 2.0,
    # 'lines.markersize': 8,
    # 'xtick.direction': 'in',
    # 'ytick.direction': 'in',
    # 'xtick.major.size': 6,
    # 'ytick.major.size': 6,
    # 'xtick.minor.size': 3,
    # 'ytick.minor.size': 3,
    # 'xtick.major.width': 1.2,
    # 'ytick.major.width': 1.2,
    # 'xtick.minor.width': 1.0,
    # 'ytick.minor.width': 1.0,
    'legend.frameon': False,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'figure.figsize': (FIG_WIDTH, FIG_WIDTH//2),
    'pdf.fonttype': 42,  # TrueType fonts for compatibility
    'ps.fonttype': 42,
    'text.usetex': False,  # Set to True if you want LaTeX rendering and have it installed
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif', 'serif'],
    'mathtext.fontset': 'stix',
})

def panel_label(ax, label):
    ax.text(
        -0.2, 1.2, label,
        transform=ax.transAxes,      # use axes coordinates
        fontsize=12,
        verticalalignment='top',
        horizontalalignment='left'
    )



def plot_pca_subplot(ax, data_dict, title, cb=False, y_label=True, plot_var=None, s=None):
    if s is None:
        s = 50
    markers = ['o', 'v', '*', 'v', '^', 'p', 'h', '8', 'X', 'd']
    if plot_var is not None:
        h = plot_var
    else:
        h = data_dict['hidden_states'][-1].cpu().numpy()
    loc_y = data_dict['loc_y']
    action_taken = data_dict['action_taken']
    corridor = data_dict['corridor']
    cond = abs(action_taken) <= 1
    # If loc_y is 2D, color by the first dimension
    color = loc_y[:, 0] if loc_y.ndim > 1 else loc_y
    color = color[cond]
    corridor = corridor[cond]
    h = h[cond]
    # PCA to 2D
    from sklearn.decomposition import PCA
    h_pca = PCA(n_components=2).fit_transform(h)
    for cor, marker in zip(np.unique(corridor), markers):
        mask = corridor == cor
        sc = ax.scatter(
            h_pca[mask, 0], h_pca[mask, 1], c=color[mask], cmap='PuOr',
            s=s, alpha=1, edgecolor='none', marker=marker
        )
    ax.set_title(title, pad=10)
    ax.axis('equal')
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_xlabel('PC 1')
    if y_label:
        ax.set_ylabel('PC 2')
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.5)
    plt.gca().set_axisbelow(True)
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)
    if cb:
        cbar = plt.colorbar(sc, ax=ax)
        cbar.ax.set_yticklabels([])  # Remove colorbar ticks
        cbar.set_label('Location')
        cbar.ax.set_yticks([])
        cbar.outline.set_linewidth(0.4) 