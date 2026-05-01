from tqdm import tqdm
import matplotlib.pyplot as plt
from data_modules import create_data
import numpy as np
from run_sim import Config
from predictive_coding_network import train_deep_pcn
import pickle
import multiprocessing as mp
import torch


_WORKER_PINNED_DEVICE: str | None = None

# X: shape (n_samples, n_features)
# y:
#   - classification: integer labels, shape (n_samples,)
#   - regression: continuous targets, shape (n_samples, output_dim)
def _resolve_device(device_name: str) -> str:
    if not torch.cuda.is_available():
        return "cpu"
    if device_name.startswith("cuda"):
        return device_name
    return "cpu"


def _init_pool_worker(gpu_devices: tuple[str, ...]) -> None:
    global _WORKER_PINNED_DEVICE
    if not gpu_devices:
        _WORKER_PINNED_DEVICE = "cpu"
        return

    identity = mp.current_process()._identity
    worker_idx = identity[0] - 1 if identity else 0
    selected = gpu_devices[worker_idx % len(gpu_devices)]
    _WORKER_PINNED_DEVICE = _resolve_device(selected)


def run_single_sim(task: tuple[int, int, int, int]) -> tuple[int, int, float, float, float]:
    i_seed, seed, i_max_move, max_move = task

    C = Config()
    C.length_corridors = [20]
    C.max_move = max_move
    C.one_hot_inputs = True
    C.one_hot_actions = True
    C.print_progress = False
    C.seed = seed

    X, y, corridor, loc_X, loc_y, action_taken, dim_l, input_size, output_size, n_actions = create_data(C)
    y = np.argmax(y, axis=1)

    L = 5
    model, history = train_deep_pcn(
        X,
        y,
        L=L,  # input -> hidden1 -> hidden2 -> output
        hidden_dims=[256] * (L - 1),  # must have length L-1, or pass a single int
        infer_lr=0.05,  # state inference step size
        weight_lr=0.2,  # local synaptic learning rate
        inference_steps=30,  # number of relaxation steps per batch
        activation="relu",
        epochs=1000,
        batch_size=X.shape[0],
        seed=seed,
        verbose=False,
        device=_WORKER_PINNED_DEVICE or "cpu",
    )

    # PCA of the last hidden layer (falls back to output layer if no hidden layer exists)
    r2_score, pr_score = model.plot_pca_last_layer(
        X[abs(action_taken) <= 1],
        y[abs(action_taken) <= 1],
        save_path=f"last_layer_pca_seed.png",
        title=f"PCA seed={seed}, max_move={max_move}",
        show=False,
    )
    train_score = model.score(X, y)
    return i_max_move, i_seed, float(r2_score), float(pr_score), float(train_score)


def save_results(result_path: str, r2_l: np.ndarray, pr_l: np.ndarray, train_score_l: np.ndarray, max_move_l: np.ndarray, n_seeds: int) -> None:
    results = {
        "r2_l": r2_l,
        "pr_l": pr_l,
        "train_score_l": train_score_l,
        "max_move_l": max_move_l,
        "n_seeds": n_seeds,
    }
    with open(result_path, "wb") as f:
        pickle.dump(results, f)


def main() -> None:
    max_move_l = np.arange(1, 20)
    n_seeds = 10
    result_path = "result_arrays.pkl"
    gpu_devices = ("cuda:4", "cuda:5", "cuda:6", "cuda:7")

    r2_l = np.zeros((len(max_move_l), n_seeds))
    pr_l = np.zeros((len(max_move_l), n_seeds))
    train_score_l = np.zeros((len(max_move_l), n_seeds))

    tasks: list[tuple[int, int, int, int]] = []
    for i_seed, seed in enumerate(range(n_seeds)):
        for i_max_move, max_move in enumerate(max_move_l):
            tasks.append((i_seed, seed, i_max_move, int(max_move)))

    n_workers = 8
    with mp.Pool(processes=n_workers, initializer=_init_pool_worker, initargs=(gpu_devices,)) as pool:
        for i_max_move, i_seed, r2_score, pr_score, train_score in tqdm(
            pool.imap_unordered(run_single_sim, tasks),
            total=len(tasks),
            desc=f"Running {len(tasks)} simulations",
        ):
            r2_l[i_max_move, i_seed] = r2_score
            pr_l[i_max_move, i_seed] = pr_score
            train_score_l[i_max_move, i_seed] = train_score
            save_results(result_path, r2_l, pr_l, train_score_l, max_move_l, n_seeds)

    train_score_mean = train_score_l.mean(axis=1)
    train_score_std = train_score_l.std(axis=1)
    r2_mean = r2_l.mean(axis=1)
    r2_std = r2_l.std(axis=1)
    pr_mean = pr_l.mean(axis=1)
    pr_std = pr_l.std(axis=1)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    line_train = ax1.errorbar(
        max_move_l, train_score_mean, yerr=train_score_std, label="Train Score", marker="o", capsize=3
    )
    line_r2 = ax1.errorbar(max_move_l, r2_mean, yerr=r2_std, label="R^2", marker="o", capsize=3)
    line_pr = ax2.errorbar(max_move_l, pr_mean, yerr=pr_std, label="PR", color="red", marker="o", capsize=3)

    ax1.set_xlabel("Max Move")
    ax1.set_ylabel("Score")
    ax2.set_ylabel("PR")
    ax1.set_title("Score vs Max Move")
    ax1.legend([line_train, line_r2, line_pr], ["Train Score", "R^2", "PR"], loc="best")
    fig.savefig("score_vs_max_move.png")
    plt.show()


if __name__ == "__main__":
    main()


