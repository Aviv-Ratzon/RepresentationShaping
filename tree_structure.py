import argparse
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
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
            # action = 3
            # next_state = state_curr
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
            y_seq.append(self.states_in[state_next - 1])
            loc_y_seq.append(state_next)
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
        dropout_p: float = 0.0,
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
        self.dropout = nn.Dropout(p=dropout_p)
 
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
                h[layer] = self.dropout(h[layer])
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
    plot1_state_distance: np.ndarray
    plot1_hidden_distance: np.ndarray
    plot2_state_distance: np.ndarray
    plot2_hidden_distance: np.ndarray
 
 
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

    h_f_mean = np.stack([h_f[loc_f==i].mean(0) for i in np.unique(loc_f)])
    loc_f_mean = np.unique(loc_f)
    hidden_d_mean = pdist(h_f_mean, metric="euclidean").astype(np.float64, copy=False)
    tree_d_mean = _condensed_tree_distances(state_distance_matrix, loc_f_mean)
    plot2 = float(spearmanr(hidden_d_mean, tree_d_mean).correlation)
 
    return plot1, plot2


def compute_plot_distance_data(
    h_np: np.ndarray,
    loc_y: np.ndarray,
    state_distance_matrix: np.ndarray,
    *,
    max_points: int,
    sample_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return raw and aggregated distance relations used for plot1/plot2."""
    hidden_d = pdist(h_np, metric="euclidean").astype(np.float64, copy=False)
    state_d = _condensed_tree_distances(state_distance_matrix, loc_y)

    if max_points > 0 and hidden_d.size > max_points:
        rng = np.random.default_rng(sample_seed)
        idx = rng.choice(hidden_d.size, size=max_points, replace=False)
        hidden_d = hidden_d[idx]
        state_d = state_d[idx]

    uniq = np.unique(state_d)
    mean_hidden_by_d = np.array([hidden_d[state_d == val].mean() for val in uniq], dtype=np.float64)
    return state_d, hidden_d, uniq, mean_hidden_by_d
 
 
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
    max_scatter_points: int,
    weight_decay: float,
    l1_lambda: float,
    activity_l2_lambda: float,
    dropout_p: float,
    grad_clip_norm: float,
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
        dropout_p=dropout_p,
        bias=True,
    ).to(device)
 
    criterion = nn.CrossEntropyLoss()
    # Don't touch the hard-set learning rate
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-1 if n_layers == 1 else 1e-4,
        weight_decay=weight_decay,
    )
 
    y_var = float(y_t.var().detach().cpu().item())
    if y_var == 0 or isinstance(criterion, nn.CrossEntropyLoss):
        y_var = 1.0
 
    model.train()
    loss_hist = torch.empty((epochs,), device=device, dtype=torch.float32)
    for ep in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        outputs, hidden_train = model(X_t)
        base_loss = criterion(outputs.reshape(-1, outputs.size(-1)), y_idx.reshape(-1))

        reg_loss = torch.zeros((), device=device)
        if l1_lambda > 0:
            reg_loss = reg_loss + l1_lambda * sum(p.abs().sum() for p in model.parameters())
        if activity_l2_lambda > 0:
            reg_loss = reg_loss + activity_l2_lambda * hidden_train.pow(2).mean()

        train_loss = base_loss + reg_loss
        train_loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        loss_hist[ep] = base_loss.detach()
    model.eval()
    n_states = y_t.shape[2]
    a_0_ind = -1
    X_test = torch.zeros((n_states, 1, X_t.shape[2]), dtype=torch.float32, device=device)
    for i in range(n_states):
        X_test[i, 0, i] = 1
    X_test[:, 0, a_0_ind] = 1
    with torch.no_grad():
        outputs_train_eval, _ = model(X_t)
        loss = criterion(outputs_train_eval.reshape(-1, outputs_train_eval.size(-1)), y_idx.reshape(-1))
        acc = (outputs_train_eval.argmax(dim=-1) == y_idx).float().mean().item()

        outputs, hidden_states = model(X_test)
 
        # Only first step, matching your original script.
        h_np = hidden_states[:, 0, :].detach().cpu().numpy()
        loc_first = np.arange(1, n_states+1)
 
    plot1, plot2 = compute_plot_metrics(
        h_np=h_np,
        loc_y=loc_first,
        state_distance_matrix=state_distance_matrix,
    )
    plot1_x, plot1_y, plot2_x, plot2_y = compute_plot_distance_data(
        h_np=h_np,
        loc_y=loc_first,
        state_distance_matrix=state_distance_matrix,
        max_points=max_scatter_points,
        sample_seed=model_seed + 7,
    )
 
    loss_curve = (loss_hist / float(y_var)).detach().cpu().numpy()
    metrics = Metrics(
        loss_norm=float(loss) / float(y_var),
        accuracy=float(acc),
        plot1_spearman_hidden_vs_tree=plot1,
        plot2_spearman_distance_vs_mean_hidden=plot2,
    )
    return TrainOutput(
        metrics=metrics,
        loss_curve=loss_curve,
        plot1_state_distance=plot1_x,
        plot1_hidden_distance=plot1_y,
        plot2_state_distance=plot2_x,
        plot2_hidden_distance=plot2_y,
    )
 
 
@dataclass(frozen=True)
class SweepConfig:
    d: int = 4
    # If empty, defaults to k=1..3*(d-1) (matching your original sweep).
    k_values: tuple[int, ...] = ()
    n_seeds: int = 5
    base_seed: int = 0
    hidden_size: int = 512
    epochs: int = 100
    lr: float = 1e-3
    num_workers: int = 4
    gpu_ids: tuple[int, ...] = tuple(range(4,8))
    max_scatter_points_per_job: int = 3000
    # Sweep modes: (label, n_layers, weight_decay, l1_lambda, activity_l2_lambda, dropout_p)
    # Default starter set:
    # 1) all off
    # 2) only n_layers "on"
    # 3) only weight_decay "on"
    # 4) only l1 "on"
    # 5) only activity_l2 "on"
    # 6) only dropout "on"
    reg_mode_names: tuple[str, ...] = (
        "none",
        "5_layers",
        # "w_decay",
        # "l1",
        # "l2",
        # "dropout",
        # "3_layers",
    )
    reg_mode_n_layers: tuple[int, ...] = (1, 5) #, 1, 1, 1, 1, 3)
    reg_mode_weight_decay: tuple[float, ...] = (0.0, 0.0) #, 1e-4, 0.0, 0.0, 0.0, 0.0)
    reg_mode_l1_lambda: tuple[float, ...] = (0.0, 0.0) #, 0.0, 1e-7, 0.0, 0.0, 0.0)
    reg_mode_activity_l2_lambda: tuple[float, ...] = (0.0, 0.0) #, 0.0, 0.0, 1e-4, 0.0, 0.0)
    reg_mode_dropout_p: tuple[float, ...] = (0.0, 0.0) #, 0.0, 0.0, 0.0, 0.1, 0.0)
    num_hidden_output_layers: int = 1
    grad_clip_norm: float = 1.0
    output_dir: str = "tree_structure"
    device: Optional[str] = None


def _validate_mode_config(cfg: SweepConfig) -> None:
    n = len(cfg.reg_mode_names)
    lengths = {
        "reg_mode_n_layers": len(cfg.reg_mode_n_layers),
        "reg_mode_weight_decay": len(cfg.reg_mode_weight_decay),
        "reg_mode_l1_lambda": len(cfg.reg_mode_l1_lambda),
        "reg_mode_activity_l2_lambda": len(cfg.reg_mode_activity_l2_lambda),
        "reg_mode_dropout_p": len(cfg.reg_mode_dropout_p),
    }
    bad = [f"{k}={v}" for k, v in lengths.items() if v != n]
    if bad:
        raise ValueError(
            "All mode tuples must have same length as reg_mode_names "
            f"(n={n}); got: {', '.join(bad)}"
        )


def _run_single_job_task(
    *,
    seed_idx: int,
    seed: int,
    k_idx: int,
    k: int,
    reg_idx: int,
    cfg: SweepConfig,
    gpu_id: Optional[int],
) -> tuple[int, int, int, float, float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run a single (seed, k, n_layers) job."""
    state_dm = compute_state_distance_matrix(cfg.d)
    n_rollouts_per_state = 2**cfg.d - 1

    if cfg.device is not None:
        device = torch.device(cfg.device)
    elif torch.cuda.is_available() and gpu_id is not None:
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    rng = np.random.default_rng(seed + 10_000 * int(k))
    env = TreeEnv(cfg.d, int(k), state_dm)
    data = generate_dataset(env, n_rollouts_per_state=n_rollouts_per_state, rng=rng)
    n_layers = int(cfg.reg_mode_n_layers[reg_idx])
    weight_decay = float(cfg.reg_mode_weight_decay[reg_idx])
    l1_lambda = float(cfg.reg_mode_l1_lambda[reg_idx])
    activity_l2_lambda = float(cfg.reg_mode_activity_l2_lambda[reg_idx])
    dropout_p = float(cfg.reg_mode_dropout_p[reg_idx])
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
        model_seed=seed + 1_000 * int(n_layers) + 100_000 * int(k) + 1_000_000 * int(reg_idx),
        max_scatter_points=cfg.max_scatter_points_per_job,
        weight_decay=weight_decay,
        l1_lambda=l1_lambda,
        activity_l2_lambda=activity_l2_lambda,
        dropout_p=dropout_p,
        grad_clip_norm=float(cfg.grad_clip_norm),
    )
    return (
        seed_idx,
        k_idx,
        reg_idx,
        out.metrics.loss_norm,
        out.metrics.accuracy,
        out.metrics.plot1_spearman_hidden_vs_tree,
        out.metrics.plot2_spearman_distance_vs_mean_hidden,
        out.loss_curve.astype(np.float32, copy=False),
        out.plot1_state_distance,
        out.plot1_hidden_distance,
        out.plot2_state_distance,
        out.plot2_hidden_distance,
    )
 
 
def run_sweep(cfg: SweepConfig) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)
    _validate_mode_config(cfg)
 
    if cfg.device is not None:
        print(f"Using explicit device: {cfg.device}")
    elif torch.cuda.is_available():
        print(f"Using multiprocessing with random GPU assignment from: {cfg.gpu_ids}")
    else:
        print("CUDA not available, running workers on CPU.")
 
    k_values = cfg.k_values if len(cfg.k_values) > 0 else tuple(range(1, 6 * (cfg.d - 1) + 1))
 
    K = len(k_values)
    R = len(cfg.reg_mode_names)
    S = cfg.n_seeds
 
    loss = np.zeros((R, S, K), dtype=np.float64)
    accuracy = np.zeros((R, S, K), dtype=np.float64)
    plot1 = np.zeros((R, S, K), dtype=np.float64)
    plot2 = np.zeros((R, S, K), dtype=np.float64)
    loss_curves = np.zeros((R, S, K, cfg.epochs), dtype=np.float32)
    plot1_scatter_x = [[[] for _ in range(R)] for _ in range(K)]
    plot1_scatter_y = [[[] for _ in range(R)] for _ in range(K)]
    plot2_scatter_x = [[[] for _ in range(R)] for _ in range(K)]
    plot2_scatter_y = [[[] for _ in range(R)] for _ in range(K)]
 
    seeds = [cfg.base_seed + i for i in range(cfg.n_seeds)]
    if cfg.device is None and torch.cuda.is_available():
        n_available = torch.cuda.device_count()
        gpu_ids = tuple(g for g in cfg.gpu_ids if 0 <= g < n_available)
        if len(gpu_ids) == 0:
            raise ValueError(f"No valid GPU IDs from {cfg.gpu_ids}; available are 0..{n_available-1}")
    else:
        gpu_ids = (None,)

    jobs = []
    for s_idx, seed in enumerate(seeds):
        for k_idx, k in enumerate(k_values):
            for reg_idx in range(R):
                jobs.append((s_idx, seed, k_idx, int(k), reg_idx))

    assign_rng = np.random.default_rng(cfg.base_seed + 99_999)
    assigned_gpu_per_job = assign_rng.choice(gpu_ids, size=len(jobs), replace=True)

    max_workers = max(1, min(int(cfg.num_workers), len(jobs)))
    ctx = mp.get_context("spawn")

    # #test single task
    # _run_single_job_task(
    #     seed_idx=0,
    #     seed=0,
    #     k_idx=0,
    #     k=3,
    #     reg_idx=0,
    #     cfg=cfg,
    #     gpu_id=None,
    # )
    # exit()
    print(f"Running {len(jobs)} jobs with {max_workers} workers")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        futures = []
        for job_idx, (s_idx, seed, k_idx, k, reg_idx) in enumerate(jobs):
            futures.append(
                ex.submit(
                    _run_single_job_task,
                    seed_idx=s_idx,
                    seed=seed,
                    k_idx=k_idx,
                    k=k,
                    reg_idx=reg_idx,
                    cfg=cfg,
                    gpu_id=None if assigned_gpu_per_job[job_idx] is None else int(assigned_gpu_per_job[job_idx]),
                )
            )

        for fut in tqdm(as_completed(futures), total=len(jobs), desc="Jobs"):
            (
                s_idx,
                k_idx,
                reg_idx,
                loss_v,
                acc_v,
                p1_v,
                p2_v,
                curve_v,
                p1x,
                p1y,
                p2x,
                p2y,
            ) = fut.result()
            loss[reg_idx, s_idx, k_idx] = loss_v
            accuracy[reg_idx, s_idx, k_idx] = acc_v
            plot1[reg_idx, s_idx, k_idx] = p1_v
            plot2[reg_idx, s_idx, k_idx] = p2_v
            loss_curves[reg_idx, s_idx, k_idx, :] = curve_v
            plot1_scatter_x[k_idx][reg_idx].append(p1x)
            plot1_scatter_y[k_idx][reg_idx].append(p1y)
            plot2_scatter_x[k_idx][reg_idx].append(p2x)
            plot2_scatter_y[k_idx][reg_idx].append(p2y)
 
    np.savez(
        os.path.join(cfg.output_dir, "results.npz"),
        k_values=np.asarray(k_values, dtype=np.int64),
        reg_mode_names=np.asarray(cfg.reg_mode_names),
        reg_mode_n_layers=np.asarray(cfg.reg_mode_n_layers, dtype=np.int64),
        reg_mode_weight_decay=np.asarray(cfg.reg_mode_weight_decay, dtype=np.float64),
        reg_mode_l1_lambda=np.asarray(cfg.reg_mode_l1_lambda, dtype=np.float64),
        reg_mode_activity_l2_lambda=np.asarray(cfg.reg_mode_activity_l2_lambda, dtype=np.float64),
        reg_mode_dropout_p=np.asarray(cfg.reg_mode_dropout_p, dtype=np.float64),
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
 
    for reg_idx, mode_name in enumerate(cfg.reg_mode_names):
        label = (
            f"{mode_name}"
        )
        ax_loss.errorbar(k_arr, loss_mean[reg_idx], yerr=loss_sem[reg_idx], marker="o", capsize=3, label=label)
        ax_acc.errorbar(k_arr, acc_mean[reg_idx], yerr=acc_sem[reg_idx], marker="o", capsize=3, label=label)
        ax_p1.errorbar(k_arr, plot1_mean[reg_idx], yerr=plot1_sem[reg_idx], marker="o", capsize=3, label=label)
        ax_p2.errorbar(k_arr, plot2_mean[reg_idx], yerr=plot2_sem[reg_idx], marker="o", capsize=3, label=label)
 
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

    # Scatter figure: rows are k values, columns are [plot1, plot2],
    # with x=state distance, y=hidden distance, colored by n_layers.
    fig_scatter, axs_scatter = plt.subplots(
        nrows=K,
        ncols=2,
        figsize=(10, max(2.4 * K, 4.0)),
        squeeze=False,
    )
    colors = plt.cm.tab10(np.linspace(0, 1, max(2, R)))

    for k_idx, k in enumerate(k_values):
        ax1 = axs_scatter[k_idx, 0]
        ax2 = axs_scatter[k_idx, 1]

        for reg_idx, mode_name in enumerate(cfg.reg_mode_names):
            p1x = np.concatenate(plot1_scatter_x[k_idx][reg_idx]) if len(plot1_scatter_x[k_idx][reg_idx]) > 0 else np.array([])
            p1y = np.concatenate(plot1_scatter_y[k_idx][reg_idx]) if len(plot1_scatter_y[k_idx][reg_idx]) > 0 else np.array([])
            p2x = np.concatenate(plot2_scatter_x[k_idx][reg_idx]) if len(plot2_scatter_x[k_idx][reg_idx]) > 0 else np.array([])
            p2y = np.concatenate(plot2_scatter_y[k_idx][reg_idx]) if len(plot2_scatter_y[k_idx][reg_idx]) > 0 else np.array([])
            label = (
                f"{mode_name}"
                if k_idx == 0
                else None
            )
            ax1.scatter(p1x, p1y, s=6, alpha=0.35, color=colors[reg_idx], label=label)
            ax2.scatter(p2x, p2y, s=10, alpha=0.65, color=colors[reg_idx], label=label)

        ax1.set_ylabel(f"k={k}\nhidden distance")
        ax2.set_ylabel("hidden distance")
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)

        if k_idx == 0:
            ax1.set_title("PLOT 1: pairwise distances")
            ax2.set_title("PLOT 2: mean hidden per state distance")
            ax1.legend(loc="best", fontsize=8)

    axs_scatter[-1, 0].set_xlabel("state distance")
    axs_scatter[-1, 1].set_xlabel("state distance")
    fig_scatter.tight_layout()
    scatter_path = os.path.join(cfg.output_dir, "plot_scatter.png")
    fig_scatter.savefig(scatter_path, dpi=200)
    print(f"Saved scatter figure to: {scatter_path}")

    # Loss curves over epochs: one subplot per (n_layers, k), one line per seed.
    epochs_arr = np.arange(1, cfg.epochs + 1, dtype=np.int64)
    fig2, axs2 = plt.subplots(
        nrows=R,
        ncols=K,
        figsize=(3.2 * K, 2.4 * R),
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

    for reg_idx, mode_name in enumerate(cfg.reg_mode_names):
        for k_idx, k in enumerate(k_values):
            ax = axs2[reg_idx, k_idx]
            for s_idx, seed in enumerate(seeds):
                ax.plot(epochs_arr, loss_curves[reg_idx, s_idx, k_idx], alpha=0.8, linewidth=1.0)
            ax.set_yscale("log")
            ax.set_ylim(y_min, y_max)
            ax.grid(True, alpha=0.25)
            if reg_idx == 0:
                ax.set_title(f"k={k}")
            if k_idx == 0:
                ax.set_ylabel(f"{mode_name}\nloss")
            if reg_idx == R - 1:
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
    p.add_argument("--n-seeds", type=int, default=2)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--num-hidden-output-layers", type=int, default=3)
    p.add_argument("--epochs", type=int, default=100000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--gpu-ids", nargs="*", default=None, help="GPU IDs sampled randomly per seed task (default: 0..7).")
    p.add_argument("--mode-names", nargs="*", default=None, help="Regularization mode labels.")
    p.add_argument("--mode-n-layers", nargs="*", default=None, help="n_layers value per mode.")
    p.add_argument("--mode-weight-decay", nargs="*", default=None, help="weight_decay value per mode.")
    p.add_argument("--mode-l1-lambda", nargs="*", default=None, help="l1_lambda value per mode.")
    p.add_argument("--mode-activity-l2-lambda", nargs="*", default=None, help="activity_l2_lambda value per mode.")
    p.add_argument("--mode-dropout-p", nargs="*", default=None, help="dropout_p value per mode.")
    p.add_argument("--grad-clip-norm", type=float, default=1.0, help="Clip gradient norm; <=0 disables clipping.")
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
 
    gpu_ids = _parse_int_list(args.gpu_ids)
    if gpu_ids is None:
        gpu_ids = tuple(range(8))

    mode_names = tuple(args.mode_names) if args.mode_names is not None and len(args.mode_names) > 0 else None
    mode_n_layers = _parse_int_list(args.mode_n_layers)
    mode_weight_decay = tuple(float(v) for v in args.mode_weight_decay) if args.mode_weight_decay else None
    mode_l1_lambda = tuple(float(v) for v in args.mode_l1_lambda) if args.mode_l1_lambda else None
    mode_activity_l2_lambda = (
        tuple(float(v) for v in args.mode_activity_l2_lambda) if args.mode_activity_l2_lambda else None
    )
    mode_dropout_p = tuple(float(v) for v in args.mode_dropout_p) if args.mode_dropout_p else None

    default_modes = SweepConfig()
    reg_mode_names = default_modes.reg_mode_names if mode_names is None else mode_names
    reg_mode_n_layers = default_modes.reg_mode_n_layers if mode_n_layers is None else tuple(int(v) for v in mode_n_layers)
    reg_mode_weight_decay = (
        default_modes.reg_mode_weight_decay if mode_weight_decay is None else tuple(float(v) for v in mode_weight_decay)
    )
    reg_mode_l1_lambda = (
        default_modes.reg_mode_l1_lambda if mode_l1_lambda is None else tuple(float(v) for v in mode_l1_lambda)
    )
    reg_mode_activity_l2_lambda = (
        default_modes.reg_mode_activity_l2_lambda
        if mode_activity_l2_lambda is None
        else tuple(float(v) for v in mode_activity_l2_lambda)
    )
    reg_mode_dropout_p = default_modes.reg_mode_dropout_p if mode_dropout_p is None else tuple(float(v) for v in mode_dropout_p)
 
    cfg = SweepConfig(
        d=int(args.d),
        k_values=tuple(int(k) for k in k_values),
        n_seeds=int(args.n_seeds),
        base_seed=int(args.base_seed),
        hidden_size=int(args.hidden_size),
        num_hidden_output_layers=int(args.num_hidden_output_layers),
        epochs=int(args.epochs),
        lr=float(args.lr),
        num_workers=int(args.num_workers),
        gpu_ids=tuple(int(g) for g in gpu_ids),
        reg_mode_names=reg_mode_names,
        reg_mode_n_layers=reg_mode_n_layers,
        reg_mode_weight_decay=reg_mode_weight_decay,
        reg_mode_l1_lambda=reg_mode_l1_lambda,
        reg_mode_activity_l2_lambda=reg_mode_activity_l2_lambda,
        reg_mode_dropout_p=reg_mode_dropout_p,
        grad_clip_norm=float(args.grad_clip_norm),
        output_dir=str(args.output_dir),
        device=args.device,
    )
    run_sweep(cfg)
 
 
if __name__ == "__main__":
    main()
