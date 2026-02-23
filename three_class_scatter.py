import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def make_3class_2d(n_per_class=200, seed=0, std=0.3, centers=None):
    """
    Generate 2D data with 3 classes.

    Returns:
        X: (3*n_per_class, 2) float array
        y: (3*n_per_class,) int array in {0,1,2}
    """
    rng = np.random.default_rng(seed)

    if centers is None:
        centers = np.array(
            [
                [-1.0, -1.0],
                [1.0, -1.0],
                [0.0, np.cos(np.pi/6)],
            ],
            dtype=float,
        )
    else:
        centers = np.asarray(centers, dtype=float)
        if centers.shape != (3, 2):
            raise ValueError(f"centers must have shape (3,2); got {centers.shape}")

    X_parts = []
    y_parts = []
    for c in range(3):
        Xc = centers[c] + rng.normal(loc=0.0, scale=std, size=(n_per_class, 2))
        yc = np.full(n_per_class, c, dtype=int)
        X_parts.append(Xc)
        y_parts.append(yc)

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    return X, y


def build_deep_linear_net(input_dim=2, hidden_dim=16, num_classes=3, n_layers=10, bias=True):
    """
    Deep *linear* network: Linear -> Linear -> ... -> Linear (no activations).
    n_layers counts the number of Linear layers (including the final classifier layer).
    """
    try:
        import torch.nn as nn
    except ImportError as e:
        raise ImportError("PyTorch is required. Install it with `pip install torch`.") from e

    if n_layers < 2:
        raise ValueError("n_layers must be >= 2.")

    layers = [nn.Linear(input_dim, hidden_dim, bias=bias)]
    for _ in range(n_layers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
    layers.append(nn.Linear(hidden_dim, num_classes, bias=bias))

    net = nn.Sequential(*layers)

    # mild init for stability in deep linear stacks
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.05)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    return net


def train_deep_linear_classifier(
    X,
    y,
    hidden_dim=16,
    n_layers=10,
    lr=3e-2,
    epochs=2500,
    seed=0,
):
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as e:
        raise ImportError("PyTorch is required. Install it with `pip install torch`.") from e

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_t = torch.tensor(np.asarray(X), dtype=torch.float32, device=device)
    y_t = torch.tensor(np.asarray(y), dtype=torch.long, device=device)

    net = build_deep_linear_net(
        input_dim=2, hidden_dim=hidden_dim, num_classes=3, n_layers=n_layers, bias=True
    ).to(device)

    opt = torch.optim.SGD(net.parameters(), lr=lr)

    losses = []
    for _ in tqdm(range(epochs)):
        opt.zero_grad(set_to_none=True)
        logits = net(X_t)
        loss = F.cross_entropy(logits, y_t)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu().item()))

    with torch.no_grad():
        preds = net(X_t).argmax(dim=1)
        acc = (preds == y_t).float().mean().item()

    return net, acc, np.asarray(losses, dtype=float)


def plot_training_loss(losses, title="Training loss (cross entropy)", logy=True):
    losses = np.asarray(losses, dtype=float)
    plt.figure(figsize=(6.5, 3.5))
    plt.plot(np.arange(1, len(losses) + 1), losses, linewidth=2)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.grid(alpha=0.25)
    if logy:
        plt.yscale("log")
    plt.tight_layout()
    plt.show()


def effective_affine_from_deep_linear(net):
    """
    For a stack of affine Linear layers, compute the *effective* affine map:
        logits(x) = W_eff x + b_eff
    via product of augmented matrices.

    Returns:
        W_eff: (3, 2)
        b_eff: (3,)
    """
    try:
        import torch.nn as nn
    except ImportError as e:
        raise ImportError("PyTorch is required. Install it with `pip install torch`.") from e

    linears = [m for m in net.modules() if isinstance(m, nn.Linear)]
    if not linears:
        raise ValueError("No Linear layers found in net.")

    A_total = None  # augmented affine matrix
    for layer in linears:
        W = layer.weight.detach().cpu().numpy().astype(float)  # (out, in)
        b = (
            layer.bias.detach().cpu().numpy().astype(float)
            if layer.bias is not None
            else np.zeros(W.shape[0], dtype=float)
        )
        out_dim, in_dim = W.shape
        A = np.zeros((out_dim + 1, in_dim + 1), dtype=float)
        A[:out_dim, :in_dim] = W
        A[:out_dim, -1] = b
        A[-1, -1] = 1.0

        A_total = A if A_total is None else (A @ A_total)

    W_eff = A_total[:-1, :-1]
    b_eff = A_total[:-1, -1]
    if W_eff.shape != (3, 2):
        raise ValueError(f"Expected effective shape (3,2); got {W_eff.shape}")
    return W_eff, b_eff


def fit_svm_ovr(X, y, C=1.0):
    try:
        from sklearn.svm import LinearSVC
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required for the SVM. Install it with `pip install scikit-learn`."
        ) from e

    clf = LinearSVC(C=C, multi_class="ovr", dual="auto", random_state=0, max_iter=50_000)
    clf.fit(X, y)
    W = np.asarray(clf.coef_, dtype=float)       # (3, 2)
    b = np.asarray(clf.intercept_, dtype=float)  # (3,)
    return W, b


def plot_scatter_with_hyperplanes(
    X,
    y,
    W_svm,
    b_svm,
    W_model,
    b_model,
    title="Model vs SVM (one-vs-rest)",
):
    X = np.asarray(X)
    y = np.asarray(y)

    plt.figure(figsize=(7.0, 6.0))

    class_colors = ["tab:blue", "tab:orange", "tab:green"]

    # Fix plot limits to data range (prevents lines from rescaling axes)
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    x_pad = 0.25 * (x_max - x_min + 1e-12)
    y_pad = 0.25 * (y_max - y_min + 1e-12)
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad

    # points
    for c, color in zip([0, 1, 2], class_colors):
        m = y == c
        plt.scatter(X[m, 0], X[m, 1], s=18, alpha=0.9, c=color)

    # line x-range
    xs = np.linspace(x_min, x_max, 500)

    def draw_lines(W, b, prefix, linestyle, alpha, clip_to_ylim=False):
        for c, (w0, w1), bi in zip([0, 1, 2], W, b):
            lab = f"{prefix} {c} vs rest"
            color = class_colors[c]
            if abs(w1) < 1e-12:
                if abs(w0) < 1e-12:
                    continue
                x0 = -bi / w0
                plt.axvline(x0, linestyle=linestyle, linewidth=2, alpha=alpha, label=lab, color=color)
            else:
                ys = -(w0 * xs + bi) / w1
                if clip_to_ylim:
                    # Break the line outside the visible y-range (prevents huge segments)
                    ys = ys.copy()
                    ys[(ys < y_min) | (ys > y_max)] = np.nan
                plt.plot(xs, ys, linestyle=linestyle, linewidth=2, alpha=alpha, label=lab, color=color)

    # use a different palette for lines vs points
    draw_lines(W_svm, b_svm, prefix="SVM", linestyle="--", alpha=0.9)
    draw_lines(W_model, b_model, prefix="model", linestyle="-", alpha=0.9)
    # Generate a dense grid over the plot area
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Compute model class scores: grid_points @ W_model.T + b_model
    scores = grid_points @ np.array(W_model).T + np.array(b_model)
    pred = np.argmax(scores, axis=1)

    # Color map for the region shading (light, matching scatter points)
    from matplotlib.colors import ListedColormap
    region_cmap = ListedColormap(
        ["#b3cdfb", "#ffd8b1", "#b6e5b6"]  # lighter blue, orange, green
    )

    plt.contourf(
        xx,
        yy,
        pred.reshape(xx.shape),
        levels=[-0.5, 0.5, 1.5, 2.5],
        colors=region_cmap.colors,
        alpha=0.28,
        linewidths=0,
        zorder=0,
    )

    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(alpha=0.25)
    plt.legend(frameon=True, fontsize=9, ncols=2)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.tight_layout()
    plt.show()

def plot_singular_value_spectrum(W_model, W_svm):
    W_model = W_model / np.linalg.norm(W_model, axis=0)
    W_svm = W_svm / np.linalg.norm(W_svm, axis=0)
    U_model, S_model, V_model = np.linalg.svd(W_model)
    U_svm, S_svm, V_svm = np.linalg.svd(W_svm)
    plt.figure(figsize=(10, 5))
    plt.plot(S_model, marker='o', label='model')
    plt.plot(S_svm, marker='o', label='svm')
    plt.xlabel('Singular Value Index')
    plt.ylabel('Singular Value')
    plt.title('Singular Value Spectrum')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    X, y = make_3class_2d(n_per_class=250, seed=2, std=0.2)
    n_layers = 15
    net, acc, losses = train_deep_linear_classifier(
        X, y, hidden_dim=64, n_layers=n_layers, lr=0.05, epochs=10000000, seed=0
    )
    W_model, b_model = effective_affine_from_deep_linear(net)
    W_svm, b_svm = fit_svm_ovr(X, y, C=1.0)

    print(f"Train accuracy (deep linear): {acc:.3f}")
    plot_training_loss(losses, title="10-layer linear net: training loss", logy=True)
    plot_scatter_with_hyperplanes(
        X,
        y,
        W_svm=W_svm,
        b_svm=b_svm,
        W_model=W_model,
        b_model=b_model,
        title=f"{n_layers}-layer linear net vs linear SVM (acc={acc:.3f})",
    )

    plot_singular_value_spectrum(W_model.T, W_svm.T)