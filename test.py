import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# -----------------------------
# User-controlled parameters
# -----------------------------
L = 5  # Grid size (LxL); number of states is S=L^2
scheme = 1  # 1: N samples -> k-step end-state one-hot; 2: all L^2 -> k-step distribution
N = 1000  # Only used when scheme=1
k = 3  # Number of random moves

activation = "relu"  # "linear" or "relu"

hidden_dim = L**2 + 1
num_hidden_layers = 3  # "deep" network depth (linear network is still deep if activation="linear")

epochs = 200
batch_size = L**2
lr = 1e-3
seed = 0

# "kldiv": KL(log-softmax(logits) || target)
# "mse": MSE(softmax(logits), target)
# "mse_logits": MSE(logits, target) directly on logits vs target vectors
# "ce": cross-entropy with class index = argmax(target)
# "l1": L1(softmax(logits), target)
loss_type = "ce"


def set_seed(s: int) -> None:
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


set_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

S = L * L


def build_transition_matrix(L_: int) -> np.ndarray:
    """
    T[a, b] = P(next=b | current=a) for a simple 4-neighbor random walk
    that chooses uniformly among valid in-bounds moves.
    """
    S_ = L_ * L_
    T = np.zeros((S_, S_), dtype=np.float32)
    for r in range(L_):
        for c in range(L_):
            a = r * L_ + c
            neighbors = [r * L_ + c]
            if r > 0:
                neighbors.append((r - 1) * L_ + c)
            if r < L_ - 1:
                neighbors.append((r + 1) * L_ + c)
            if c > 0:
                neighbors.append(r * L_ + (c - 1))
            if c < L_ - 1:
                neighbors.append(r * L_ + (c + 1))

            if not neighbors:
                # Degenerate case: L_=1
                T[a, a] = 1.0
                continue

            deg = len(neighbors)
            p = 1.0 / deg
            for b in neighbors:
                T[a, b] = p
    return T


def generate_scheme_1(L_: int, N_: int, k_: int, rng: np.random.Generator):
    """
    N data points. Input = one-hot(start). Output = one-hot(end after k random moves).
    """
    S_ = L_ * L_
    start_idxs = rng.integers(0, S_, size=N_, endpoint=False, dtype=np.int64)
    X = np.eye(S_, dtype=np.float32)[start_idxs]

    end_idxs = np.empty(N_, dtype=np.int64)
    for i in range(N_):
        pos = int(start_idxs[i])
        for _ in range(k_):
            r = pos // L_
            c = pos % L_
            neighbors = [r * L_ + c]
            if r > 0:
                neighbors.append((r - 1) * L_ + c)
            if r < L_ - 1:
                neighbors.append((r + 1) * L_ + c)
            if c > 0:
                neighbors.append(r * L_ + (c - 1))
            if c < L_ - 1:
                neighbors.append(r * L_ + (c + 1))

            pos = int(rng.choice(neighbors))
        end_idxs[i] = pos

    Y = np.eye(S_, dtype=np.float32)[end_idxs]
    return X, Y, end_idxs


def generate_scheme_2(L_: int, k_: int):
    """
    L^2 data points. Input = one-hot(start).
    Output = probability distribution of position after k-step random walk.
    """
    S_ = L_ * L_
    X = np.eye(S_, dtype=np.float32)  # one-hot for each start state

    T = build_transition_matrix(L_)
    # Since X is identity, evolving all start states at once is:
    #   D0 = I, Dk = D0 @ T^k = T^k
    D = X.copy()  # shape [S, S], rows are start distributions over current state
    for _ in range(k_):
        D = D @ T
    Y = D.astype(np.float32)
    return X, Y


class DeepMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim_: int, out_dim: int, num_hidden_layers_: int, act: str):
        super().__init__()
        if act not in ("linear", "relu"):
            raise ValueError("activation must be 'linear' or 'relu'")

        layers = []
        prev = in_dim
        for i in range(num_hidden_layers_):
            lin = nn.Linear(prev, hidden_dim_, bias=False)
            # Small weight init for hidden layers
            nn.init.normal_(lin.weight, mean=0.0, std=0.01)
            # nn.init.constant_(lin.bias, 0.0)
            layers.append(lin)
            if act == "relu":
                layers.append(nn.ReLU())
            prev = hidden_dim_

        self.feature_net = nn.Sequential(*layers)  # last hidden activations live here

        self.head = nn.Linear(hidden_dim_, out_dim, bias=False)
        # Large weight init for output layer
        nn.init.normal_(self.head.weight, mean=0.0, std=1.0)
        # nn.init.constant_(self.head.bias, 0.0)

    def forward(self, x: torch.Tensor):
        h = self.feature_net(x)
        logits = self.head(h)
        return logits, h


rng = np.random.default_rng(seed)

if scheme == 1:
    X_np, Y_np, end_idxs_np = generate_scheme_1(L, N, k, rng)
else:
    X_np, Y_np = generate_scheme_2(L, k)
    end_idxs_np = None

X = torch.from_numpy(X_np).to(device)
Y = torch.from_numpy(Y_np).to(device)


model = DeepMLP(
    in_dim=S,
    hidden_dim_=hidden_dim,
    out_dim=S,
    num_hidden_layers_=num_hidden_layers,
    act=activation,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

VALID_LOSS_TYPES = ("kldiv", "mse", "mse_logits", "ce", "l1")


def compute_loss(logits: torch.Tensor, targets: torch.Tensor, loss_type_: str) -> torch.Tensor:
    if loss_type_ not in VALID_LOSS_TYPES:
        raise ValueError(f"loss_type must be one of {VALID_LOSS_TYPES}, got {loss_type_!r}")

    if loss_type_ == "kldiv":
        log_probs = F.log_softmax(logits, dim=1)
        return F.kl_div(log_probs, targets, reduction="batchmean")

    if loss_type_ == "mse":
        probs = F.softmax(logits, dim=1)
        return F.mse_loss(logits, targets)

    if loss_type_ == "mse_logits":
        return F.mse_loss(logits, targets)

    if loss_type_ == "l1":
        probs = F.softmax(logits, dim=1)
        return F.l1_loss(probs, targets)

    # cross_entropy
    class_idx = targets.argmax(dim=1)
    return F.cross_entropy(logits, class_idx)


def accuracy_one_hot(pred_logits: torch.Tensor, target_probs: torch.Tensor) -> float:
    pred = pred_logits.argmax(dim=1)
    target = target_probs.argmax(dim=1)
    return (pred == target).float().mean().item()


dataset = torch.utils.data.TensorDataset(X, Y)
loader = torch.utils.data.DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)

losses = []
accuracies = []

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0

    for xb, yb in loader:
        logits, _ = model(xb)
        loss = compute_loss(logits, yb, loss_type)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * len(xb)

    epoch_loss = running_loss / len(loader.dataset)
    losses.append(epoch_loss)

    if scheme == 1:
        model.eval()
        with torch.no_grad():
            logits_all, _ = model(X)
            acc = accuracy_one_hot(logits_all, Y)
        accuracies.append(acc)

    if epoch == 1 or epoch % max(1, epochs // 10) == 0:
        if scheme == 1:
            print(f"epoch {epoch:4d}/{epochs}: loss={epoch_loss:.6f}, acc={accuracies[-1]:.4f}")
        else:
            print(f"epoch {epoch:4d}/{epochs}: loss={epoch_loss:.6f}")


# -----------------------------
# Plotting
# -----------------------------
plt.figure()
plt.plot(np.arange(1, epochs + 1), losses, linewidth=2)
plt.xlabel("Epoch")
plt.ylabel(f"Loss ({loss_type})")
plt.title(f"Loss over task (L={L}, scheme={scheme}, k={k}, act={activation}, loss={loss_type})")
plt.grid(True, alpha=0.3)
plt.yscale('log')


if scheme == 1:
    plt.figure()
    plt.plot(np.arange(1, epochs + 1), accuracies, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (argmax vs end-state)")
    plt.title("Accuracy over task")
    plt.grid(True, alpha=0.3)


def pca_2d(data: np.ndarray) -> np.ndarray:
    """Manual 2D PCA via SVD (rows = samples)."""
    centered = data - data.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ Vt[:2].T


def scatter_pca(ax, coords: np.ndarray, color_idx: np.ndarray, color_label: str, title: str):
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=color_idx, s=50, cmap="viridis", alpha=0.9)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label(color_label)
    return sc


# PCA of last hidden activations and targets, colored by grid row / column
model.eval()
with torch.no_grad():
    X_grid = torch.eye(S, dtype=torch.float32, device=device)
    _, h = model(X_grid)
    h_np = h.detach().cpu().numpy()

H_pca = pca_2d(h_np)
Y_pca = pca_2d(Y_np)

grid_state_idx = np.arange(S, dtype=np.int64)
h_row_idx = grid_state_idx // L
h_col_idx = grid_state_idx % L

if scheme == 2:
    y_state_idx = grid_state_idx
else:
    y_state_idx = X_np.argmax(axis=1).astype(np.int64)

y_row_idx = y_state_idx // L
y_col_idx = y_state_idx % L

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

scatter_pca(axes[0, 0], H_pca, h_row_idx, "Row index (0..L-1)", "Hidden activations — row")
scatter_pca(axes[0, 1], H_pca, h_col_idx, "Column index (0..L-1)", "Hidden activations — column")
scatter_pca(axes[1, 0], Y_pca, y_row_idx, "Row index (0..L-1)", "Targets Y — row")
scatter_pca(axes[1, 1], Y_pca, y_col_idx, "Column index (0..L-1)", "Targets Y — column")

fig.suptitle("PCA: hidden activations (top) and targets Y (bottom)")
fig.tight_layout()

plt.show()


# def calc_OLS(X, y):
#     X = X.astype(np.float64)
#     y = y.astype(np.float64)
#     beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
#     return beta

# fig, axs = plt.subplots(1, 3, figsize=(10, 5))
# for k in [1, 10]:
#     X, Y = generate_scheme_2(20, k)
#     OLS = calc_OLS(X, Y)
#     U, s_vals, V = np.linalg.svd(Y, full_matrices=False)
#     h = Y #  U @ np.diag(s_vals)
#     h_centered = h - h.mean(axis=0, keepdims=True)

#     # 3. Perform SVD
#     U, s, Vt = np.linalg.svd(h_centered, full_matrices=False)

#     PCs_2 = Vt[:2].T
#     # shape: (n_features, 2)

#     # 4. Projection onto first 2 PCs
#     h_pca = h_centered @ PCs_2

#     axs[0].plot(s_vals, marker='o', label=f'k={k}')
#     axs[1].scatter(h_pca[:, 0], h_pca[:, 1], marker='o', label=f'k={k}')
#     axs[2].plot(V[0, :], marker='o', label=f'k={k}')
# plt.legend()
# plt.show()
