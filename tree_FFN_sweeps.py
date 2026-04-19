import argparse
import hashlib
import json
import multiprocessing as mp
import os
from dataclasses import asdict, dataclass, replace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


UP_ACTION = 0
DOWN_LEFT_ACTION = 1
DOWN_RIGHT_ACTION = 2
STAY_ACTION = 3

_WORKER_PINNED_DEVICE: str | None = None


def path_between_indices(a: int, b: int) -> list[int]:
    if a < 1 or b < 1:
        raise ValueError("Indices must be positive integers.")

    path_a = []
    x = a
    while x >= 1:
        path_a.append(x)
        x //= 2

    path_b = []
    x = b
    while x >= 1:
        path_b.append(x)
        x //= 2

    path_a.reverse()
    path_b.reverse()

    i = 0
    while i < len(path_a) and i < len(path_b) and path_a[i] == path_b[i]:
        i += 1

    up_moves = [UP_ACTION] * (len(path_a) - i)
    down_moves = []
    for node in path_b[i:]:
        if node % 2 == 0:
            down_moves.append(DOWN_LEFT_ACTION)
        else:
            down_moves.append(DOWN_RIGHT_ACTION)

    return up_moves + down_moves


class Tree:
    def __init__(self, d: int, k: int):
        self.d = d
        self.k = k
        self.states = np.arange(1, 2**d)
        self.n_states = len(self.states)
        self.actions = [UP_ACTION, DOWN_LEFT_ACTION, DOWN_RIGHT_ACTION]
        self.states_in = np.eye(self.n_states)
        self.actions_in = np.eye(len(self.actions))

    def walk(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_features = len(self.states) + len(self.actions) * self.k
        X = []
        y = []
        loc_X = []
        loc_y = []
        action_taken = []

        for state_start in self.states:
            for target_state in self.states:
                path = path_between_indices(state_start, target_state)
                if len(path) > self.k:
                    continue

                state_curr = state_start
                actions_in = []
                actions = []
                for action in path:
                    state_curr, action_taken_curr = self.take_action(state_curr, action)
                    actions_in.append(self.actions_in[action_taken_curr])
                    actions.append(action_taken_curr)

                X_seq = self.states_in[state_start - 1]
                if actions_in:
                    X_seq = np.concatenate([X_seq, np.concatenate(actions_in)])
                X_seq = np.pad(X_seq, (0, n_features - len(X_seq)), constant_values=0)
                y_seq = self.states_in[state_curr - 1]

                action_taken_seq = np.pad(
                    np.array(actions, dtype=int),
                    (0, self.k - len(actions)),
                    constant_values=STAY_ACTION,
                )

                X.append(X_seq)
                y.append(y_seq)
                loc_X.append(state_start)
                loc_y.append(state_curr)
                action_taken.append(action_taken_seq)

        return (
            np.stack(X),
            np.stack(y),
            np.array(loc_X),
            np.array(loc_y),
            np.stack(action_taken),
        )

    def take_action(self, state_curr: int, action: int) -> tuple[int, int]:
        next_state = self.move(state_curr, action)
        if next_state not in self.states:
            raise ValueError(
                f"Next state {next_state} not in states, start state: {state_curr}, action: {action}"
            )
        return next_state, action

    @staticmethod
    def move(state: int, action: int) -> int:
        if action == UP_ACTION:
            return state // 2
        if action == DOWN_LEFT_ACTION:
            return state * 2
        if action == DOWN_RIGHT_ACTION:
            return state * 2 + 1
        if action == STAY_ACTION:
            return state
        raise ValueError(f"Unsupported action: {action}")


class DeepNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        init_scale: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            layer = nn.Linear(in_dim, hidden_size)
            nn.init.xavier_normal_(layer.weight, gain=init_scale)
            nn.init.zeros_(layer.bias)
            self.hidden_layers.append(layer)
            self.dropouts.append(nn.Dropout(dropout))

        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = x
        for layer, dropout in zip(self.hidden_layers, self.dropouts):
            hidden = torch.relu(layer(hidden))
            hidden = dropout(hidden)
        output = self.output_layer(hidden)
        return output, hidden


@dataclass(frozen=True)
class TrainConfig:
    d: int = 7
    hidden_size: int = 64
    num_layers: int = 3
    init_scale: float = 1.0
    l2_reg: float = 0.0
    l1_reg: float = 0.0
    dropout: float = 0.0
    epochs: int = 10000
    lr: float = 1e-2
    lr_min: float = 1e-4
    lr_max: float = 5e-2
    lr_increase_factor: float = 1.2
    lr_decrease_factor: float = 0.5
    lr_increase_patience: int = 25
    lr_decrease_patience: int = 50
    lr_improvement_tol: float = 1e-5
    device: str = "cpu"
    gpu_devices: tuple[str, ...] = ("cuda:4", "cuda:5", "cuda:6", "cuda:7")
    base_seed: int = 0
    n_seeds: int = 1


@dataclass(frozen=True)
class SweepTask:
    sweep_name: str
    sweep_value: float
    k: int
    seed_idx: int
    config: TrainConfig


def _stable_int_seed(*values: Any) -> int:
    text = "::".join(map(str, values))
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def _build_dataset(d: int, k: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tree = Tree(d=d, k=k)
    X, y, loc_X, loc_y, _ = tree.walk()

    omit_loc = loc_y.max()
    omit_inds = np.where((loc_y == omit_loc) | (loc_X == omit_loc))[0]
    omit_size = max(1, len(omit_inds) // 2)
    omit_inds = rng.choice(omit_inds, size=omit_size, replace=False)

    X_test = X[omit_inds]
    y_test = y[omit_inds]

    X_train = np.delete(X, omit_inds, axis=0)
    y_train = np.delete(y, omit_inds, axis=0)

    return X_train, y_train, X_test, y_test


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def _init_pool_worker(gpu_devices: tuple[str, ...]) -> None:
    global _WORKER_PINNED_DEVICE
    if not gpu_devices:
        _WORKER_PINNED_DEVICE = None
        return
    identity = mp.current_process()._identity
    worker_idx = identity[0] - 1 if identity else 0
    _WORKER_PINNED_DEVICE = gpu_devices[worker_idx % len(gpu_devices)]


def _select_task_device(cfg: TrainConfig, rng: np.random.Generator) -> torch.device:
    if cfg.device == "random_cuda":
        if _WORKER_PINNED_DEVICE is not None:
            return _resolve_device(_WORKER_PINNED_DEVICE)
        if not cfg.gpu_devices:
            return _resolve_device("cpu")
        selected_device = str(rng.choice(cfg.gpu_devices))
        return _resolve_device(selected_device)
    return _resolve_device(cfg.device)


def run_single_task(task: SweepTask) -> dict[str, Any]:
    torch.set_num_threads(1)

    cfg = task.config
    seed = _stable_int_seed(cfg.base_seed, task.sweep_name, task.sweep_value, task.k, task.seed_idx)
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    X_train, y_train, X_test, y_test = _build_dataset(d=cfg.d, k=task.k, rng=rng)
    device = _select_task_device(cfg, rng)

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32, device=device)

    model = DeepNet(
        input_size=X_train.shape[1],
        hidden_size=cfg.hidden_size,
        output_size=y_train.shape[1],
        num_layers=cfg.num_layers,
        init_scale=cfg.init_scale,
        dropout=cfg.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr/(2**cfg.num_layers), weight_decay=cfg.l2_reg)

    train_loss = 0.0
    train_acc = 0.0
    loss_curve: list[float] = []
    best_loss = float("inf")
    good_epochs = 0
    bad_epochs = 0
    current_lr = float(cfg.lr)
    for _ in range(cfg.epochs):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(X_train_t)
        loss = criterion(logits, y_train_t.argmax(dim=-1))

        if cfg.l1_reg > 0:
            l1_penalty = sum(param.abs().sum() for param in model.parameters())
            loss = loss + cfg.l1_reg * l1_penalty

        loss.backward()
        optimizer.step()

        train_loss = float(loss.item())
        loss_curve.append(train_loss)
        train_acc = float((logits.argmax(dim=-1) == y_train_t.argmax(dim=-1)).float().mean().item())

        # Adaptive LR: raise when consistently improving, lower when plateauing/worsening.
        if train_loss < best_loss - cfg.lr_improvement_tol:
            best_loss = train_loss
            good_epochs += 1
            bad_epochs = 0
            if good_epochs >= cfg.lr_increase_patience:
                new_lr = min(current_lr * cfg.lr_increase_factor, cfg.lr_max)
                if new_lr > current_lr:
                    current_lr = new_lr
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = current_lr
                good_epochs = 0
        else:
            bad_epochs += 1
            good_epochs = 0
            if bad_epochs >= cfg.lr_decrease_patience:
                new_lr = max(current_lr * cfg.lr_decrease_factor, cfg.lr_min)
                if new_lr < current_lr:
                    current_lr = new_lr
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = current_lr
                bad_epochs = 0

    model.eval()
    with torch.no_grad():
        logits_test, _ = model(X_test_t)
        test_loss = float(criterion(logits_test, y_test_t.argmax(dim=-1)).item())
        test_acc = float((logits_test.argmax(dim=-1) == y_test_t.argmax(dim=-1)).float().mean().item())

    return {
        "sweep_name": task.sweep_name,
        "sweep_value": task.sweep_value,
        "k": task.k,
        "seed_idx": task.seed_idx,
        "train_accuracy": train_acc,
        "train_loss": train_loss,
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "final_lr": current_lr,
        "loss_curve": loss_curve,
    }


def _make_sweep_tasks(
    sweep_name: str,
    sweep_values: list[float],
    base_config: TrainConfig,
) -> list[SweepTask]:
    tasks: list[SweepTask] = []
    for value in sweep_values:
        cfg = replace(base_config)
        if sweep_name == "num_layers":
            cfg = replace(cfg, num_layers=int(value))
        elif sweep_name == "init_scale":
            cfg = replace(cfg, init_scale=float(value))
        elif sweep_name == "l2_reg":
            cfg = replace(cfg, l2_reg=float(value))
        elif sweep_name == "l1_reg":
            cfg = replace(cfg, l1_reg=float(value))
        elif sweep_name == "dropout":
            cfg = replace(cfg, dropout=float(value))
        elif sweep_name == "hidden_size":
            cfg = replace(cfg, hidden_size=int(value))
        elif sweep_name == "d":
            cfg = replace(cfg, d=int(value))
        else:
            raise ValueError(f"Unsupported sweep: {sweep_name}")

        k_values = list(range(1, 2 * (cfg.d - 1) + 1))
        tasks.extend(
            SweepTask(
                sweep_name=sweep_name,
                sweep_value=value,
                k=k,
                seed_idx=seed_idx,
                config=cfg,
            )
            for seed_idx in range(max(1, cfg.n_seeds))
            for k in k_values
        )
    return tasks


def _plot_single_sweep(
    sweep_name: str,
    sweep_values: list[float],
    results: list[dict[str, Any]],
    output_dir: str,
) -> str:
    def _mean_std_by_k(rows: list[dict[str, Any]], metric: str) -> tuple[list[int], np.ndarray, np.ndarray]:
        k_values = sorted({int(row["k"]) for row in rows})
        means: list[float] = []
        stds: list[float] = []
        for k in k_values:
            metric_values = np.array([float(row[metric]) for row in rows if int(row["k"]) == k], dtype=float)
            means.append(float(metric_values.mean()))
            stds.append(float(metric_values.std(ddof=0)))
        return k_values, np.array(means, dtype=float), np.array(stds, dtype=float)

    def _bounded_yerr(means: np.ndarray, stds: np.ndarray, lo: float, hi: float) -> np.ndarray:
        lower = np.clip(means - stds, lo, hi)
        upper = np.clip(means + stds, lo, hi)
        return np.vstack([means - lower, upper - means])

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8))

    for value in sweep_values:
        rows = [row for row in results if row["sweep_name"] == sweep_name and row["sweep_value"] == value]
        rows = sorted(rows, key=lambda row: row["k"])
        if not rows:
            continue

        k_l, train_acc_mean, train_acc_std = _mean_std_by_k(rows, "train_accuracy")
        _, train_loss_mean, train_loss_std = _mean_std_by_k(rows, "train_loss")
        _, test_acc_mean, test_acc_std = _mean_std_by_k(rows, "test_accuracy")
        _, test_loss_mean, test_loss_std = _mean_std_by_k(rows, "test_loss")

        label = f"{sweep_name}: {value}"
        ax1.errorbar(
            k_l,
            train_acc_mean,
            yerr=_bounded_yerr(train_acc_mean, train_acc_std, 0.0, 1.0),
            marker="o",
            capsize=3,
            alpha=0.9,
        )
        ax2.errorbar(
            k_l,
            train_loss_mean,
            yerr=_bounded_yerr(train_loss_mean, train_loss_std, 1e-12, float("inf")),
            marker="o",
            capsize=3,
            alpha=0.9,
        )
        ax3.errorbar(
            k_l,
            test_acc_mean,
            yerr=_bounded_yerr(test_acc_mean, test_acc_std, 0.0, 1.0),
            marker="o",
            capsize=3,
            label=label,
            alpha=0.9,
        )
        ax4.errorbar(
            k_l,
            test_loss_mean,
            yerr=_bounded_yerr(test_loss_mean, test_loss_std, 1e-12, float("inf")),
            marker="o",
            capsize=3,
            alpha=0.9,
        )

    ax2.set_yscale("log")
    ax4.set_yscale("log")
    ax1.set_ylim(-0.1, 1.1)
    ax3.set_ylim(-0.1, 1.1)

    ax1.set_title("Train accuracy")
    ax2.set_title("Train loss")
    ax3.set_title("Test accuracy")
    ax4.set_title("Test loss")
    ax1.set_ylabel("Train")
    ax3.set_ylabel("Test")
    ax3.legend(fontsize=9)
    fig.suptitle(f"Sweep: {sweep_name}")
    fig.tight_layout()

    output_path = os.path.join(output_dir, f"sweep_{sweep_name}.png")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_loss_curves_for_sweep(
    sweep_name: str,
    sweep_values: list[float],
    results: list[dict[str, Any]],
    output_dir: str,
) -> str:
    fig, ax = plt.subplots(figsize=(11, 8))

    for value in sweep_values:
        rows = [row for row in results if row["sweep_name"] == sweep_name and row["sweep_value"] == value]
        rows = sorted(rows, key=lambda row: row["k"])
        for row in rows:
            loss_curve = row.get("loss_curve", [])
            if not loss_curve:
                continue
            epochs = np.arange(1, len(loss_curve) + 1)
            label = f"{sweep_name}={value}, k={row['k']}"
            ax.plot(epochs, loss_curve, alpha=0.35, linewidth=1.0, label=label)

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train loss")
    ax.set_title(f"Loss over epochs - {sweep_name} sweep (all runs)")

    handles, labels = ax.get_legend_handles_labels()
    max_legend_entries = 25
    if len(handles) <= max_legend_entries:
        ax.legend(fontsize=7, ncol=2)
    else:
        ax.legend(handles[:max_legend_entries], labels[:max_legend_entries], fontsize=7, ncol=2, title="First 25 runs")

    fig.tight_layout()
    output_path = os.path.join(output_dir, f"sweep_{sweep_name}_loss_over_epochs.png")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def run_all_sweeps(
    base_config: TrainConfig,
    sweeps: dict[str, list[float]],
    output_dir: str,
    n_workers: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for sweep_name, sweep_values in sweeps.items():
        tasks = _make_sweep_tasks(sweep_name, sweep_values, base_config)
        desc = f"{sweep_name} ({len(tasks)} jobs)"
        print(f"\nStarting sweep '{sweep_name}' with {len(sweep_values)} values and {len(tasks)} tasks")

        if n_workers == 1:
            results = [run_single_task(task) for task in tqdm(tasks, desc=desc)]
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(
                processes=n_workers,
                initializer=_init_pool_worker,
                initargs=(base_config.gpu_devices,),
            ) as pool:
                results = list(tqdm(pool.imap_unordered(run_single_task, tasks), total=len(tasks), desc=desc))

        results = sorted(results, key=lambda r: (r["sweep_value"], r["k"]))
        plot_path = _plot_single_sweep(sweep_name, sweep_values, results, output_dir)
        loss_plot_path = _plot_loss_curves_for_sweep(sweep_name, sweep_values, results, output_dir)
        results_path = os.path.join(output_dir, f"sweep_{sweep_name}.json")
        # with open(results_path, "w", encoding="utf-8") as f:
        #     json.dump(
        #         {
        #             "base_config": asdict(base_config),
        #             "sweep_name": sweep_name,
        #             "sweep_values": sweep_values,
        #             "results": results,
        #         },
        #         f,
        #         indent=2,
        #     )
        print(f"Saved figure: {plot_path}")
        print(f"Saved loss-curves figure: {loss_plot_path}")
        print(f"Saved metrics: {results_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel FFN sweeps on the tree task.")
    parser.add_argument("--output-dir", type=str, default="test_tree_sweeps", help="Where to save figures and metrics.")
    parser.add_argument("--epochs", type=int, default=100000, help="Training epochs per task.")
    parser.add_argument("--lr", type=float, default=1e-1, help="Learning rate.")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(mp.cpu_count() - 1, 8)),
        help="Number of multiprocessing workers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="random_cuda",
        help="cpu, auto, random_cuda, or fixed cuda device string.",
    )
    parser.add_argument(
        "--gpu-devices",
        type=str,
        default="cuda:4,cuda:5,cuda:6,cuda:7",
        help="Comma-separated CUDA devices used when --device=random_cuda.",
    )
    parser.add_argument("--base-seed", type=int, default=0, help="Global seed for reproducibility.")
    parser.add_argument("--n-seeds", type=int, default=20, help="Number of random seeds per (sweep value, k).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpu_devices = tuple(device.strip() for device in args.gpu_devices.split(",") if device.strip())
    base_config = TrainConfig(
        d=6,
        hidden_size=512,
        num_layers=1,
        init_scale=1.0,
        l2_reg=0.0,
        l1_reg=0.0,
        dropout=0.0,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        gpu_devices=gpu_devices,
        base_seed=args.base_seed,
        n_seeds=max(1, args.n_seeds),
    )

    sweeps = {
        "num_layers": [1, 2, 3],
        # "init_scale": [1.0, 0.5, 0.25, 0.1],
        # "l2_reg": [0.0, 1e-5, 1e-4, 1e-3],
        # "l1_reg": [0.0, 1e-7, 1e-6, 1e-5],
        # "dropout": [0.0, 0.1, 0.2, 0.4],
        # "hidden_size": [16, 32, 64, 128, 256],
        # "d": [3, 4, 5, 6],
    }

    run_all_sweeps(
        base_config=base_config,
        sweeps=sweeps,
        output_dir=args.output_dir,
        n_workers=max(1, args.workers),
    )


if __name__ == "__main__":
    main()
