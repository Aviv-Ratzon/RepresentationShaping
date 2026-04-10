import argparse
import os
from dataclasses import dataclass
from typing import Iterable, Optional
 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from tqdm import tqdm
import matplotlib.pyplot as plt
 
 
def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
 
 
def path_between_indices(a: int, b: int) -> list[int]:
    """
    Return the shortest path from node a to node b in a binary tree
    indexed like:
        1
      2   3
     4 5 6 7
 
    Actions:
        0 = up
        1 = down left
        2 = down right
    """
    if a < 1 or b < 1:
        raise ValueError("Indices must be positive integers.")
 
    path_a: list[int] = []
    x = a
    while x >= 1:
        path_a.append(x)
        x //= 2
 
    path_b: list[int] = []
    x = b
    while x >= 1:
        path_b.append(x)
        x //= 2
 
    path_a.reverse()
    path_b.reverse()
 
    i = 0
    while i < len(path_a) and i < len(path_b) and path_a[i] == path_b[i]:
        i += 1
 
    lca_depth = i  # path_a[i-1] is the LCA (if i>0)
    up_moves = [0] * (len(path_a) - lca_depth)
 
    down_moves: list[int] = []
    for node in path_b[lca_depth:]:
        down_moves.append(1 if node % 2 == 0 else 2)
 
    return up_moves + down_moves
 
 
def depth_zero_indexed(i: int) -> int:
    """Compute depth of node i in 0-indexed heap layout (root=0)."""
    d = 0
    while i > 0:
        i = (i - 1) // 2
        d += 1
    return d
 
 
def lca_zero_indexed(i: int, j: int) -> int:
    """Lowest common ancestor for 0-indexed heap layout."""
    di, dj = depth_zero_indexed(i), depth_zero_indexed(j)
    while di > dj:
        i = (i - 1) // 2
        di -= 1
    while dj > di:
        j = (j - 1) // 2
        dj -= 1
    while i != j:
        i = (i - 1) // 2
        j = (j - 1) // 2
    return i
 
 
def tree_distance(i_one_indexed: int, j_one_indexed: int) -> int:
    """Distance between nodes in 1-indexed binary tree node IDs."""
    i = i_one_indexed - 1
    j = j_one_indexed - 1
    di = depth_zero_indexed(i)
    dj = depth_zero_indexed(j)
    ancestor = lca_zero_indexed(i, j)
    da = depth_zero_indexed(ancestor)
    return di + dj - 2 * da
 
 
def compute_state_distance_matrix(d: int) -> np.ndarray:
    """Pairwise tree distances over states 1..(2**d-1)."""
    states = np.arange(1, 2**d)
    n = len(states)
    dm = np.zeros((n, n), dtype=np.int16)
    for i in range(n):
        for j in range(n):
            dm[i, j] = tree_distance(i + 1, j + 1)
    return dm
 
 
@dataclass(frozen=True)
class Dataset:
    X: np.ndarray  # (N, k, input_size)
    y: np.ndarray  # (N, k, n_states) one-hot labels
    loc_y: np.ndarray  # (N, k) one-indexed state IDs
    action_taken: np.ndarray  # (N, k) in {0,1,2,3}
 
 
class TreeEnv:
    def __init__(self, d: int, k: int, state_distance_matrix: np.ndarray):
        self.d = d
        self.k = k
        self.states = np.arange(1, 2**d)  # 1-indexed
        self.n_states = len(self.states)
        self.actions = np.array([0, 1, 2, 3], dtype=np.int64)
        self.states_in = np.eye(self.n_states, dtype=np.float32)
        self.actions_in = np.eye(len(self.actions), dtype=np.float32)
        self.state_distance_matrix = state_distance_matrix.astype(np.int16, copy=False)
 
    def move(self, state: int, action: int) -> int:
        if action == 0:
            return state // 2
        if action == 1:
            return state * 2
        if action == 2:
            return state * 2 + 1
        if action == 3:
            return state
        raise ValueError(f"Invalid action: {action}")
 
    def take_action(self, state_curr: int, action: Optional[int], rng: np.random.Generator) -> tuple[int, int]:
        if action is not None:
            next_state = self.move(state_curr, action)
        else:
            next_state = -1
            action = -1
            while next_state not in self.states:
                action = int(rng.choice(self.actions))
                next_state = self.move(state_curr, action)
        if next_state not in self.states:
            raise ValueError(
                f"Next state {next_state} not in states, start={state_curr}, action={action}"
            )
        return next_state, int(action)
 
    def walk(self, state_start: int, i: int, rng: np.random.Generator) -> tuple[list[np.ndarray], list[np.ndarray], list[int], list[int]]:
        """
        Deterministic target selection (like your original `i%len(neighbors)`),
        but stochastic filler actions if path is shorter than k.
        """
        neighbors = np.where(self.state_distance_matrix[state_start - 1] <= self.k)[0]  # 0-indexed
        target_state = int(neighbors[i % len(neighbors)] + 1)
        path = path_between_indices(state_start, target_state)
 
        X_seq: list[np.ndarray] = []
        y_seq: list[np.ndarray] = []
        loc_y_seq: list[int] = []
        action_taken_seq: list[int] = []
 
        state_curr = state_start
        for t in range(self.k):
            action_hint = path[t] if t < len(path) else None
            state_next, action = self.take_action(state_curr, action_hint, rng)
 
            x_t = np.concatenate(
                [(t == 0) * self.states_in[state_curr - 1], self.actions_in[action]],
                axis=0,
            ).astype(np.float32, copy=False)
            X_seq.append(x_t)
            y_seq.append(self.states_in[state_curr - 1])
            loc_y_seq.append(state_curr)
            action_taken_seq.append(action)
            state_curr = state_next
 
        return X_seq, y_seq, loc_y_seq, action_taken_seq
 
 
def generate_dataset(env: TreeEnv, n_rollouts_per_state: int, rng: np.random.Generator) -> Dataset:
    X: list[list[np.ndarray]] = []
    y: list[list[np.ndarray]] = []
    loc_y: list[list[int]] = []
    action_taken: list[list[int]] = []
 
    for state in env.states:
        for i in range(n_rollouts_per_state):
            X_seq, y_seq, loc_y_seq, action_taken_seq = env.walk(int(state), int(i), rng)
            X.append(X_seq)
            y.append(y_seq)
            loc_y.append(loc_y_seq)
            action_taken.append(action_taken_seq)
 
    return Dataset(
        X=np.asarray(X, dtype=np.float32),
        y=np.asarray(y, dtype=np.float32),
        loc_y=np.asarray(loc_y, dtype=np.int64),
        action_taken=np.asarray(action_taken, dtype=np.int64),
    )
 
 
class LinearRNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        num_hidden_output_layers: int = 3,
        bias: bool = True,
    ):
        super().__init__()
 
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
 
        self.W_ih = nn.ParameterList()
        self.biases = nn.ParameterList() if bias else None
 
        for layer in range(num_layers):
            in_dim = input_size if layer == 0 else hidden_size
            self.W_ih.append(
                nn.Parameter(torch.randn(hidden_size, in_dim) * (0.9 / np.sqrt(in_dim)))
            )
            if bias:
                self.biases.append(nn.Parameter(torch.zeros(hidden_size)))
 
        hidden_output_layers = nn.ModuleList()
        for _ in range(num_hidden_output_layers):
            hidden_output_layers.append(nn.Linear(hidden_size, hidden_size, bias=bias))
            hidden_output_layers.append(nn.ReLU())
        self.hidden_output_layers = nn.Sequential(*hidden_output_layers)
        self.output_layer = nn.Linear(hidden_size, output_size, bias=bias)
 
    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        batch_first=True
 
        x:  (B, T, input_size)
        h0: (num_layers, B, hidden_size)
        """
        B, T, _ = x.shape
        if h0 is None:
            h = [x.new_zeros(B, self.hidden_size) for _ in range(self.num_layers)]
        else:
            h = [h0[layer] for layer in range(self.num_layers)]
 
        outputs: list[torch.Tensor] = []
        hidden_states: list[torch.Tensor] = []
 
        for t in range(T):
            input_t = x[:, t, :]
            for layer in range(self.num_layers):
                prev_h = h[layer]
                linear = input_t @ self.W_ih[layer].T + prev_h
                if self.biases is not None:
                    linear = linear + self.biases[layer]
                h[layer] = F.relu(linear)
                input_t = h[layer]
            # h[-1] = self.hidden_output_layers(h[-1])
            hidden_states.append(h[-1])
            outputs.append(self.output_layer(h[-1]))
 
        return torch.stack(outputs, dim=1), torch.stack(hidden_states, dim=1)
 
 
@dataclass(frozen=True)
class Metrics:
    loss_norm: float
    accuracy: float
    plot1_spearman_hidden_vs_tree: float
    plot2_spearman_distance_vs_mean_hidden: float


@dataclass(frozen=True)
class TrainOutput:
    metrics: Metrics
    loss_curve: np.ndarray  # (epochs,)
 
 
def _sem(x: np.ndarray, axis: int = 0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[axis]
    res = x.std(axis=axis, ddof=1) / np.sqrt(n)
    res[np.isnan(res)] = 0
    return res
 
 
def _condensed_tree_distances(state_dm: np.ndarray, loc_y: np.ndarray) -> np.ndarray:
    """
    loc_y: (N,) one-indexed state IDs.
    Returns condensed upper-triangle distances (like scipy.pdist output).
    """
    loc0 = loc_y.astype(np.int64) - 1
    iu = np.triu_indices(loc0.shape[0], k=1)
    return state_dm[loc0[iu[0]], loc0[iu[1]]].astype(np.float64, copy=False)
 
 
def compute_plot_metrics(
    h_np: np.ndarray,
    loc_y: np.ndarray,
    action_taken: np.ndarray,
    state_distance_matrix: np.ndarray,
) -> tuple[float, float]:
    """
    Replicates your PLOT 1 / PLOT 2 logic, but plotted vs k:
 
    - PLOT 1: Spearman(hidden_distance, tree_distance) after filtering action_taken<=1.
    - PLOT 2: Spearman(unique_tree_distance, mean_hidden_distance_per_tree_distance).
    """
    h_f = h_np
    loc_f = loc_y
 
    hidden_d = pdist(h_f, metric="euclidean").astype(np.float64, copy=False)
    tree_d = _condensed_tree_distances(state_distance_matrix, loc_f)
 
    plot1 = float(spearmanr(hidden_d, tree_d).correlation)
 
    uniq = np.unique(tree_d)
    mean_hidden_by_d = np.array([hidden_d[tree_d == val].mean() for val in uniq], dtype=np.float64)
    plot2 = float(spearmanr(uniq, mean_hidden_by_d).correlation)
 
    return plot1, plot2
 
 
def train_and_evaluate(
    X: np.ndarray,
    y_onehot: np.ndarray,
    loc_y: np.ndarray,
    action_taken: np.ndarray,
    *,
    n_layers: int,
    hidden_size: int,
    num_hidden_output_layers: int,
    epochs: int,
    lr: float,
    device: torch.device,
    state_distance_matrix: np.ndarray,
    model_seed: int,
) -> TrainOutput:
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_onehot, dtype=torch.float32, device=device)
    y_idx = y_t.argmax(dim=-1)
 
    seed_everything(model_seed)
    model = LinearRNN(
        input_size=X.shape[2],
        hidden_size=hidden_size,
        output_size=y_onehot.shape[2],
        num_layers=n_layers,
        num_hidden_output_layers=num_hidden_output_layers,
        bias=True,
    ).to(device)
 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3 if n_layers == 1 else 1e-5)
 
    y_var = float(y_t.var().detach().cpu().item())
    if y_var == 0 or isinstance(criterion, nn.CrossEntropyLoss):
        y_var = 1.0
 
    model.train()
    loss_hist = torch.empty((epochs,), device=device, dtype=torch.float32)
    for ep in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        outputs, _ = model(X_t)
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), y_idx.reshape(-1))
        loss.backward()
        optimizer.step()
        loss_hist[ep] = loss.detach()
 
    model.eval()
    with torch.no_grad():
        outputs, hidden_states = model(X_t)
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), y_idx.reshape(-1))
        acc = (outputs.argmax(dim=-1) == y_idx).float().mean().item()
 
        # Only first step, matching your original script.
        h_np = hidden_states[:, 0, :].detach().cpu().numpy()
        loc_first = loc_y[:, 0]
        action_first = action_taken[:, 0]
 
    plot1, plot2 = compute_plot_metrics(
        h_np=h_np,
        loc_y=loc_first,
        action_taken=action_first,
        state_distance_matrix=state_distance_matrix,
    )
 
    loss_curve = (loss_hist / float(y_var)).detach().cpu().numpy()
    metrics = Metrics(
        loss_norm=float(loss.detach().cpu().item()) / float(y_var),
        accuracy=float(acc),
        plot1_spearman_hidden_vs_tree=plot1,
        plot2_spearman_distance_vs_mean_hidden=plot2,
    )
    return TrainOutput(metrics=metrics, loss_curve=loss_curve)
 
 
@dataclass(frozen=True)
class SweepConfig:
    d: int = 4
    # If empty, defaults to k=1..3*(d-1) (matching your original sweep).
    k_values: tuple[int, ...] = ()
    n_layers_values: tuple[int, ...] = (1, 5)
    n_seeds: int = 5
    base_seed: int = 0
    hidden_size: int = 512
    num_hidden_output_layers: int = 1
    epochs: int = 100
    lr: float = 1e-3
    output_dir: str = "tree_structure"
    device: Optional[str] = None
 
 
def run_sweep(cfg: SweepConfig) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)
 
    device = torch.device(cfg.device) if cfg.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
 
    k_values = cfg.k_values if len(cfg.k_values) > 0 else tuple(range(1, 3 * (cfg.d - 1) + 1))
    state_dm = compute_state_distance_matrix(cfg.d)
    n_rollouts_per_state = 2**cfg.d - 1
 
    K = len(k_values)
    L = len(cfg.n_layers_values)
    S = cfg.n_seeds
 
    loss = np.zeros((L, S, K), dtype=np.float64)
    accuracy = np.zeros((L, S, K), dtype=np.float64)
    plot1 = np.zeros((L, S, K), dtype=np.float64)
    plot2 = np.zeros((L, S, K), dtype=np.float64)
    loss_curves = np.zeros((L, S, K, cfg.epochs), dtype=np.float32)
 
    seeds = [cfg.base_seed + i for i in range(cfg.n_seeds)]
 
    for s_idx, seed in enumerate(tqdm(seeds, desc="Seeds")):
        for k_idx, k in enumerate(tqdm(k_values, desc="k", leave=False)):
            # Make dataset randomness depend on (seed, k), but independent of model init.
            rng = np.random.default_rng(seed + 10_000 * int(k))
            np.random.seed(seed + 10_000 * int(k))
            torch.manual_seed(seed + 10_000 * int(k))
            env = TreeEnv(cfg.d, int(k), state_dm)
            data = generate_dataset(env, n_rollouts_per_state=n_rollouts_per_state, rng=rng)
 
            for l_idx, n_layers in enumerate(tqdm(cfg.n_layers_values, desc="n_layers", leave=False)):
                
                out = train_and_evaluate(
                    data.X,
                    data.y,
                    data.loc_y,
                    data.action_taken,
                    n_layers=int(n_layers),
                    hidden_size=cfg.hidden_size,
                    num_hidden_output_layers=cfg.num_hidden_output_layers,
                    epochs=cfg.epochs,
                    lr=cfg.lr,
                    device=device,
                    state_distance_matrix=state_dm,
                    model_seed=seed + 1_000 * int(n_layers),
                )
 
                loss[l_idx, s_idx, k_idx] = out.metrics.loss_norm
                accuracy[l_idx, s_idx, k_idx] = out.metrics.accuracy
                plot1[l_idx, s_idx, k_idx] = out.metrics.plot1_spearman_hidden_vs_tree
                plot2[l_idx, s_idx, k_idx] = out.metrics.plot2_spearman_distance_vs_mean_hidden
                loss_curves[l_idx, s_idx, k_idx, :] = out.loss_curve.astype(np.float32, copy=False)
 
    np.savez(
        os.path.join(cfg.output_dir, "results.npz"),
        k_values=np.asarray(k_values, dtype=np.int64),
        n_layers_values=np.asarray(cfg.n_layers_values, dtype=np.int64),
        loss=loss,
        accuracy=accuracy,
        plot1=plot1,
        plot2=plot2,
        seeds=np.asarray(seeds, dtype=np.int64),
        d=np.asarray(cfg.d, dtype=np.int64),
        hidden_size=np.asarray(cfg.hidden_size, dtype=np.int64),
        epochs=np.asarray(cfg.epochs, dtype=np.int64),
        lr=np.asarray(cfg.lr, dtype=np.float64),
    )
 
    k_arr = np.asarray(k_values, dtype=np.int64)
    loss_mean, loss_sem = loss.mean(axis=1), _sem(loss, axis=1)
    acc_mean, acc_sem = accuracy.mean(axis=1), _sem(accuracy, axis=1)
    plot1_mean, plot1_sem = plot1.mean(axis=1), _sem(plot1, axis=1)
    plot2_mean, plot2_sem = plot2.mean(axis=1), _sem(plot2, axis=1)
 
    fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    ax_loss = axs[0, 0]
    ax_acc = axs[0, 1]
    ax_p1 = axs[1, 0]
    ax_p2 = axs[1, 1]
 
    for l_idx, n_layers in enumerate(cfg.n_layers_values):
        label = f"n_layers={n_layers}"
        ax_loss.errorbar(k_arr, loss_mean[l_idx], yerr=loss_sem[l_idx], marker="o", capsize=3, label=label)
        ax_acc.errorbar(k_arr, acc_mean[l_idx], yerr=acc_sem[l_idx], marker="o", capsize=3, label=label)
        ax_p1.errorbar(k_arr, plot1_mean[l_idx], yerr=plot1_sem[l_idx], marker="o", capsize=3, label=label)
        ax_p2.errorbar(k_arr, plot2_mean[l_idx], yerr=plot2_sem[l_idx], marker="o", capsize=3, label=label)
 
    ax_loss.set_title("Loss (normalized)")
    ax_loss.set_xlabel("k")
    ax_loss.set_ylabel("loss / var(y)")
    ax_loss.set_yscale('log')
    ax_loss.grid(True, alpha=0.3)
 
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("k")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_ylim(0.0, 1.0)
    ax_acc.grid(True, alpha=0.3)

    ax_p1.set_title("PLOT 1: Spearman(hidden_dist, tree_dist)")
    ax_p1.set_xlabel("k")
    ax_p1.set_ylabel("Spearman ρ")
    ax_p1.set_ylim(-0.1, 1.0)
    ax_p1.grid(True, alpha=0.3)
 
    ax_p2.set_title("PLOT 2: Spearman(dist, mean hidden_dist)")
    ax_p2.set_xlabel("k")
    ax_p2.set_ylabel("Spearman ρ")
    ax_p2.set_ylim(-0.1, 1.0)
    ax_p2.grid(True, alpha=0.3)
 
    ax_loss.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_path = os.path.join(cfg.output_dir, "sweep.png")
    fig.savefig(out_path, dpi=200)
    print(f"Saved figure to: {out_path}")

    # Loss curves over epochs: one subplot per (n_layers, k), one line per seed.
    epochs_arr = np.arange(1, cfg.epochs + 1, dtype=np.int64)
    fig2, axs2 = plt.subplots(
        nrows=L,
        ncols=K,
        figsize=(3.2 * K, 2.4 * L),
        sharex=True,
        sharey=True,
    )
    axs2 = np.atleast_2d(axs2)
    y_min = float(np.nanmin(loss_curves))
    y_max = float(np.nanmax(loss_curves))
    if not np.isfinite(y_min) or y_min <= 0:
        y_min = 1e-8
    if not np.isfinite(y_max) or y_max <= y_min:
        y_max = max(1e-2, y_min * 10)

    for l_idx, n_layers in enumerate(cfg.n_layers_values):
        for k_idx, k in enumerate(k_values):
            ax = axs2[l_idx, k_idx]
            for s_idx, seed in enumerate(seeds):
                ax.plot(epochs_arr, loss_curves[l_idx, s_idx, k_idx], alpha=0.8, linewidth=1.0)
            ax.set_yscale("log")
            ax.set_ylim(y_min, y_max)
            ax.grid(True, alpha=0.25)
            if l_idx == 0:
                ax.set_title(f"k={k}")
            if k_idx == 0:
                ax.set_ylabel(f"n_layers={n_layers}\nloss")
            if l_idx == L - 1:
                ax.set_xlabel("epoch")

    fig2.tight_layout()
    loss_path = os.path.join(cfg.output_dir, "loss.png")
    fig2.savefig(loss_path, dpi=200)
    print(f"Saved loss curves figure to: {loss_path}")
 
 
def _parse_int_list(vals: Optional[list[str]]) -> Optional[tuple[int, ...]]:
    if vals is None or len(vals) == 0:
        return None
    return tuple(int(v) for v in vals)
 
 
def main(argv: Optional[Iterable[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Tree-structure sweep over k × seeds × n_layers.")
    p.add_argument("--d", type=int, default=4)
    p.add_argument("--k-values", nargs="*", default=None, help="Explicit k values (e.g. --k-values 1 2 3).")
    p.add_argument("--k-min", type=int, default=None)
    p.add_argument("--k-max", type=int, default=None)
    p.add_argument("--n-layers", nargs="*", default=None, help="n_layers sweep (e.g. --n-layers 1 3 5).")
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--num-hidden-output-layers", type=int, default=3)
    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", type=str, default=None, help='e.g. "cuda", "cuda:0", "cpu"')
    p.add_argument("--output-dir", type=str, default="tree_structure")
 
    args = p.parse_args(list(argv) if argv is not None else None)
 
    k_values = _parse_int_list(args.k_values)
    if k_values is None:
        if args.k_min is not None or args.k_max is not None:
            k_min = 1 if args.k_min is None else int(args.k_min)
            k_max = int(args.k_max) if args.k_max is not None else (3 * (args.d - 1) + 1)
            k_values = tuple(range(k_min, k_max + 1))
        else:
            k_values = tuple(range(1, 3 * (args.d - 1) + 1 + 0))
 
    n_layers_values = _parse_int_list(args.n_layers)
    if n_layers_values is None:
        n_layers_values = (1, 5)
 
    cfg = SweepConfig(
        d=int(args.d),
        k_values=tuple(int(k) for k in k_values),
        n_layers_values=tuple(int(n) for n in n_layers_values),
        n_seeds=int(args.n_seeds),
        base_seed=int(args.base_seed),
        hidden_size=int(args.hidden_size),
        num_hidden_output_layers=int(args.num_hidden_output_layers),
        epochs=int(args.epochs),
        lr=float(args.lr),
        output_dir=str(args.output_dir),
        device=args.device,
    )
    run_sweep(cfg)
 
 
if __name__ == "__main__":
    main()
