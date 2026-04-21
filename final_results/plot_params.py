import matplotlib as mpl

# Set global matplotlib formatting for ICLR paper compatibility
# Set global font to Times New Roman
# import seaborn as sns
# sns.set_context("paper")
# sns.set_style("whitegrid")
FIG_WIDTH = 5.5

mpl.rcParams.update({
    'font.size': 15,
    'axes.labelsize': 12,
    'axes.titlesize': 20,
    'xtick.labelsize': 8,
    'ytick.labelsize': 5,
    'legend.fontsize': 16,
    'figure.titlesize': 30,
    'axes.linewidth': 1.2,
    'lines.linewidth': 2.0,
    'lines.markersize': 8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.minor.width': 1.0,
    'ytick.minor.width': 1.0,
    'legend.frameon': False,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'figure.figsize': (6, 4),
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